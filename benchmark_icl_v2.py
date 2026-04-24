#!/usr/bin/env python3
"""
benchmark_icl_v2.py — In-Context Learning (ICL) / Few-Shot Prompting Study
===========================================================================

Research question:
  Can few-shot ICL with a balanced prompt improve HDFS anomaly detection
  precision (reduce false positives) without sacrificing recall?

Previous failures (benchmark_models_icl.py, 2026-04-10):
  - All models predicted "Normal" for every sequence → F1 = 0.000
  - Root cause: the system prompt explicitly told models to "default to Normal"
  - ICL demonstrations were never seen because the system prompt overrode them

This script fixes both issues and tests three ICL strategies:
  1. Zero-shot (balanced prompt)          — baseline, replicates larger_model run
  2. Few-shot k=4 (2 anomaly + 2 normal)  — standard balanced ICL
  3. Few-shot k=8 (4 anomaly + 4 normal)  — more demonstrations
  4. Few-shot k=4 normal-heavy (1 anomaly + 3 normal) — precision-focused ICL
     Hypothesis: showing more normal examples calibrates the model's threshold
     and reduces the FP rate (currently 14-15/15 normal sequences misclassified)

ICL design principles applied (from literature):
  - Balanced label distribution in demonstrations (Brown et al., 2020)
  - Recency bias: most informative demo placed last (Zhao et al., 2021)
  - Demonstrations drawn from a held-out pool (no data leakage)
  - Consistent format between demonstrations and query
  - Chain-of-thought style reasoning encouraged in the prompt

Models tested:
  - llama-3.1-8b-instant   (8B  — small, fast)
  - llama-3.3-70b-versatile (70B — large, reference)
"""
import os
import sys
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ── Load .env ──────────────────────────────────────────────────────────────
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k, v)

sys.path.insert(0, '.')
from aiopslab.core.framework import AIOpsLab
from aiopslab.agents.llm_agent import LLMAgent

# ── Configuration ──────────────────────────────────────────────────────────
MODELS = [
    ("groq", "llama-3.1-8b-instant",     "Llama 3.1 8B"),
    ("groq", "llama-3.3-70b-versatile",  "Llama 3.3 70B"),
]

SAMPLE_SIZE = 30   # 15 anomalous + 15 normal
MAX_NORMAL  = 15
RANDOM_SEED = 42   # reproducibility

# ── Balanced system prompt (no Normal bias) ────────────────────────────────
SYSTEM_PROMPT = """You are an expert in HDFS (Hadoop Distributed File System) anomaly detection.
Your task is to analyze log sequences and make binary classifications: Anomaly or Normal.

HDFS anomaly indicators include:
- Replication failures or under-replicated blocks
- Checksum / data integrity errors
- Datanode communication failures or timeouts
- Blocks being deleted or lost unexpectedly
- Error or Exception log entries

Normal HDFS operations include: block allocation, receiving blocks, addStoredBlock updates, pipeline setup.

Classify based on the evidence in the logs. Do not default to either label — weigh the evidence.

Respond ONLY with:
Label: [Anomaly/Normal]
Reason: [brief explanation]"""


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict]]:
    """Load dataset and split into eval sample + held-out demo pool."""
    random.seed(seed)
    lab = AIOpsLab()
    dataset = lab.load_dataset("hdfs")
    sequences = dataset.get("sequences", [])

    normal    = [s for s in sequences if not s.get("is_anomaly")]
    anomalous = [s for s in sequences if s.get("is_anomaly")]

    # Eval sample: first 15 anomalous + first 15 normal
    eval_sample = anomalous[:SAMPLE_SIZE // 2] + normal[:MAX_NORMAL]

    # Demo pool: everything else (no overlap with eval)
    demo_pool = anomalous[SAMPLE_SIZE // 2:] + normal[MAX_NORMAL:]

    n_anom = sum(1 for s in eval_sample if s.get("is_anomaly"))
    n_norm = sum(1 for s in eval_sample if not s.get("is_anomaly"))
    print(f"Eval sample : {len(eval_sample)} sequences ({n_anom} anomalous, {n_norm} normal)")
    print(f"Demo pool   : {len(demo_pool)} sequences available for ICL")
    return eval_sample, demo_pool


# ── ICL demonstration builders ─────────────────────────────────────────────

def _format_seq(seq: Dict) -> str:
    """Render a sequence as a compact log block."""
    raw_logs = seq.get("logs", [])
    if raw_logs:
        entries = [l.get("content", l.get("raw_line", "")) for l in raw_logs]
    else:
        entries = seq.get("templates") or seq.get("log_sequence") or []
    lines = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(entries[:20]))
    return (f"Block {seq.get('block_id')} "
            f"({seq.get('log_count', len(entries))} log entries):\n{lines}")


def pick_demos(pool: List[Dict], n_anomaly: int, n_normal: int) -> List[Dict]:
    """
    Select balanced demonstrations from pool.
    Most-similar-to-query heuristic is not available without embeddings,
    so we use random sampling with a fixed seed for reproducibility.
    The last demo in the list is anomalous (recency bias toward positive class).
    """
    anomalous = [s for s in pool if s.get("is_anomaly")]
    normal    = [s for s in pool if not s.get("is_anomaly")]

    chosen_anom = random.sample(anomalous, min(n_anomaly, len(anomalous)))
    chosen_norm = random.sample(normal,    min(n_normal,  len(normal)))

    # Interleave: normal first, anomalous last (recency bias)
    demos = chosen_norm + chosen_anom
    return demos


def build_icl_prompt(query_seq: Dict, demos: List[Dict]) -> str:
    """
    Build a few-shot ICL prompt.
    Structure: task description → labeled examples → query.
    The most recent demo (last in list) is anomalous to leverage recency bias.
    """
    demo_block = ""
    for demo in demos:
        label = "Anomaly" if demo.get("is_anomaly") else "Normal"
        demo_block += f"\n--- Example ---\n{_format_seq(demo)}\nLabel: {label}\n"

    query_block = _format_seq(query_seq)

    return f"""HDFS LOG SEQUENCE ANALYSIS — FEW-SHOT IN-CONTEXT LEARNING

Below are {len(demos)} labeled HDFS log sequences as reference examples, followed by a new sequence to classify.

{demo_block.strip()}

--- Query (classify this one) ---
{query_block}

ANALYSIS TASK:
Analyze the query log sequence and determine if it represents normal or anomalous behavior.

Consider these HDFS-specific patterns:
1. Block corruption (checksum failures, data integrity issues)
2. Replication issues (failed replications, under-replication)
3. Datanode failures (node crashes, communication failures)
4. Network problems (timeouts, connection issues)
5. Storage problems (disk failures, space issues)

Use the labeled examples above to calibrate your judgment.
Normal HDFS operations include: block allocation, replication, receiving blocks from multiple sources.
Only classify as ANOMALY if you see explicit errors, exceptions, or clear failure indicators.

Respond ONLY with:
Label: [Anomaly/Normal]
Reason: [brief explanation]"""


def build_zero_shot_prompt(query_seq: Dict) -> str:
    """Standard zero-shot prompt (no demonstrations)."""
    agent = LLMAgent({"provider": "groq", "model": "llama-3.1-8b-instant"})
    return agent._build_hdfs_anomaly_prompt(query_seq)


# ── Metrics ────────────────────────────────────────────────────────────────

def calc_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    tp = fp = tn = fn = 0
    for p, t in zip(predictions, ground_truth):
        pa, ta = p["predicted_anomaly"], t["actual_anomaly"]
        if   pa and ta:         tp += 1
        elif pa and not ta:     fp += 1
        elif not pa and not ta: tn += 1
        else:                   fn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1, accuracy=acc)


# ── Run helpers ────────────────────────────────────────────────────────────

def run_condition(
    provider: str,
    model_id: str,
    display_name: str,
    condition_name: str,
    sample: List[Dict],
    demo_pool: List[Dict],
    n_anomaly_demos: int,
    n_normal_demos: int,
) -> Dict:
    """
    Run one model under one ICL condition.
    n_anomaly_demos=0, n_normal_demos=0 → zero-shot.
    """
    k = n_anomaly_demos + n_normal_demos
    is_zero_shot = (k == 0)

    print(f"\n{'='*65}")
    print(f"  Model     : {display_name}")
    print(f"  Condition : {condition_name}  (k={k})")
    print(f"{'='*65}")

    agent = LLMAgent({"provider": provider, "model": model_id})

    # Build a fixed demo set for the whole run (consistent comparison)
    if not is_zero_shot:
        demos = pick_demos(demo_pool, n_anomaly_demos, n_normal_demos)
        print(f"  Demos     : {sum(1 for d in demos if d.get('is_anomaly'))} anomalous, "
              f"{sum(1 for d in demos if not d.get('is_anomaly'))} normal")

    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        true_label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"  [{i+1:2d}/{len(sample)}] Block {seq.get('block_id')} ({true_label})",
              end=" ", flush=True)
        try:
            if is_zero_shot:
                prompt = build_zero_shot_prompt(seq)
            else:
                prompt = build_icl_prompt(seq, demos)

            response = agent._call_llm(prompt, SYSTEM_PROMPT)
            result   = agent._parse_hdfs_anomaly_response(response, seq)
            pred     = result.get("predicted_anomaly", False)

            predictions.append({
                "block_id": seq.get("block_id"),
                "predicted_anomaly": pred,
                "confidence": result.get("confidence", 0),
            })
            ground_truth.append({
                "block_id": seq.get("block_id"),
                "actual_anomaly": seq.get("is_anomaly", False),
            })
            correct = (pred == seq.get("is_anomaly"))
            print("✓" if correct else "✗")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    elapsed = time.time() - t0
    metrics = calc_metrics(predictions, ground_truth)
    metrics.update(
        elapsed_sec=round(elapsed, 1),
        errors=errors,
        model=model_id,
        display_name=display_name,
        provider=provider,
        condition=condition_name,
        k=k,
    )

    print(f"\n  Precision : {metrics['precision']:.3f}  "
          f"Recall : {metrics['recall']:.3f}  "
          f"F1 : {metrics['f1']:.3f}  "
          f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  Time: {elapsed:.1f}s  Errors: {errors}")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


# ── Main ───────────────────────────────────────────────────────────────────

# ICL conditions: (name, n_anomaly_demos, n_normal_demos)
CONDITIONS = [
    ("zero-shot",              0, 0),
    ("few-shot-k4-balanced",   2, 2),
    ("few-shot-k8-balanced",   4, 4),
    ("few-shot-k4-normal-heavy", 1, 3),  # precision-focused: more normal demos
]


def main():
    sample, demo_pool = load_data()
    all_results = {}

    for provider, model_id, display_name in MODELS:
        all_results[model_id] = {"display_name": display_name, "conditions": {}}
        for cond_name, n_anom, n_norm in CONDITIONS:
            result = run_condition(
                provider, model_id, display_name,
                cond_name, sample, demo_pool,
                n_anom, n_norm,
            )
            all_results[model_id]["conditions"][cond_name] = result

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FULL COMPARISON SUMMARY  (ICL v2 — balanced prompt)")
    print(f"{'='*80}")
    header = f"{'Model + Condition':<50} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
    print(header)
    print("-" * 90)
    for model_id, data in all_results.items():
        for cond_name, res in data["conditions"].items():
            m = res["metrics"]
            tag = f"{data['display_name']} [{cond_name}]"
            print(f"{tag:<50} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
                  f"{m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4}")
        print()

    # ── Delta table (vs zero-shot baseline) ───────────────────────────────
    print(f"\n{'='*80}")
    print("ICL DELTA vs ZERO-SHOT  (positive = improvement)")
    print(f"{'='*80}")
    print(f"{'Model + Condition':<50} {'ΔPrec':>7} {'ΔRec':>7} {'ΔF1':>7} {'ΔAcc':>7}")
    print("-" * 75)
    for model_id, data in all_results.items():
        zs = data["conditions"]["zero-shot"]["metrics"]
        for cond_name, res in data["conditions"].items():
            if cond_name == "zero-shot":
                continue
            m = res["metrics"]
            tag = f"{data['display_name']} [{cond_name}]"
            print(f"{tag:<50} "
                  f"{m['precision']-zs['precision']:>+7.3f} "
                  f"{m['recall']-zs['recall']:>+7.3f} "
                  f"{m['f1']-zs['f1']:>+7.3f} "
                  f"{m['accuracy']-zs['accuracy']:>+7.3f}")
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/icl_v2_benchmark_{ts}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return out_path, all_results


if __name__ == "__main__":
    main()
