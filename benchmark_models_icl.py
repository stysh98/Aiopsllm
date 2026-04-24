#!/usr/bin/env python3
"""
Benchmark script: compare llama-3.1-8b-instant vs llama-3.3-70b-versatile
on the HDFS anomaly detection task — zero-shot vs few-shot ICL.

Extends benchmark_models.py with in-context learning (ICL) demonstrations.
"""
import os
import sys
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Load .env
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k, v)

sys.path.insert(0, '.')
from aiopslab.core.framework import AIOpsLab
from aiopslab.agents.llm_agent import LLMAgent

MODELS = [
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "llama-3.3-70b-versatile"),
]

SAMPLE_SIZE = 30   # sequences to evaluate
MAX_NORMAL = 15
FEW_SHOT_K = 4     # demonstrations per query (2 anomalous + 2 normal)


# ---------------------------------------------------------------------------
# ICL helpers
# ---------------------------------------------------------------------------

def build_few_shot_demonstrations(pool: List[Dict], k: int = FEW_SHOT_K) -> List[Dict]:
    """
    Pick k balanced demonstrations from pool (k/2 anomalous, k/2 normal).
    Demonstrations are separate from the evaluation sample.
    """
    anomalous = [s for s in pool if s.get("is_anomaly")]
    normal    = [s for s in pool if not s.get("is_anomaly")]
    half = k // 2
    demos = random.sample(anomalous, min(half, len(anomalous))) + \
            random.sample(normal,    min(half, len(normal)))
    random.shuffle(demos)
    return demos


def format_sequence_for_demo(seq: Dict) -> str:
    """Render a single sequence as a compact log block for the prompt."""
    # prefer raw log content over template list
    raw_logs = seq.get("logs", [])
    if raw_logs:
        entries = [l.get("content", l.get("raw_line", "")) for l in raw_logs]
    else:
        entries = seq.get("templates") or seq.get("log_sequence") or []
    lines = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(entries[:20]))
    return f"Block {seq.get('block_id')} ({seq.get('log_count', len(entries))} log entries):\n{lines}"


def build_icl_prompt(query_seq: Dict, demos: List[Dict]) -> str:
    """
    Build a few-shot prompt: demonstrations first, then the query.
    Most-similar demo goes last (recency bias).
    """
    demo_block = ""
    for demo in demos:
        label = "Anomaly" if demo.get("is_anomaly") else "Normal"
        demo_block += f"""
--- Example ---
{format_sequence_for_demo(demo)}
Label: {label}
"""

    query_block = format_sequence_for_demo(query_seq)

    return f"""HDFS LOG SEQUENCE ANALYSIS — FEW-SHOT

Below are labeled examples of HDFS log sequences, followed by a new sequence to classify.

{demo_block.strip()}

--- Query ---
{query_block}

ANALYSIS TASK:
Analyze the query log sequence and determine if it represents normal or anomalous behavior.

Consider these HDFS-specific patterns:
1. Block corruption (checksum failures, data integrity issues)
2. Replication issues (failed replications, under-replication)
3. Datanode failures (node crashes, communication failures)
4. Network problems (timeouts, connection issues)
5. Storage problems (disk failures, space issues)

IMPORTANT GUIDELINES:
- Normal HDFS operations include: block allocation, replication, receiving blocks from multiple sources
- Retries and multiple attempts are often normal behavior
- Only classify as ANOMALY if you see explicit errors, exceptions, or clear failure indicators
- When in doubt, classify as NORMAL

Respond with:
Label: [Anomaly/Normal]
Reason: [detailed explanation of your classification]"""


# ---------------------------------------------------------------------------
# Preserved from benchmark_models.py — unchanged
# ---------------------------------------------------------------------------

def load_sample():
    lab = AIOpsLab()
    dataset = lab.load_dataset("hdfs")
    sequences = dataset.get("sequences", [])
    normal    = [s for s in sequences if not s.get("is_anomaly")]
    anomalous = [s for s in sequences if s.get("is_anomaly")]
    # reserve a separate demo pool before slicing the eval sample
    demo_pool_anomalous = anomalous[SAMPLE_SIZE // 2:]
    demo_pool_normal    = normal[MAX_NORMAL:]
    demo_pool = demo_pool_anomalous + demo_pool_normal

    sample = anomalous[:SAMPLE_SIZE // 2] + normal[:MAX_NORMAL]
    print(f"Sample: {len(sample)} sequences "
          f"({sum(1 for s in sample if s.get('is_anomaly'))} anomalous, "
          f"{sum(1 for s in sample if not s.get('is_anomaly'))} normal)")
    print(f"Demo pool: {len(demo_pool)} sequences available for ICL demonstrations")
    return sample, demo_pool


def calc_metrics(predictions, ground_truth):
    tp = fp = tn = fn = 0
    for p, t in zip(predictions, ground_truth):
        pa, ta = p["predicted_anomaly"], t["actual_anomaly"]
        if pa and ta:         tp += 1
        elif pa and not ta:   fp += 1
        elif not pa and not ta: tn += 1
        else:                 fn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1, accuracy=acc)


# ---------------------------------------------------------------------------
# Run helpers — zero-shot (original) and few-shot (new)
# ---------------------------------------------------------------------------

def run_model_zero_shot(provider, model, sample):
    """Original zero-shot run — identical to benchmark_models.py."""
    print(f"\n{'='*60}")
    print(f"[ZERO-SHOT] Model: {model}  (provider: {provider})")
    print(f"{'='*60}")

    agent = LLMAgent({"provider": provider, "model": model})

    zero_shot_system = """You are an expert in HDFS (Hadoop Distributed File System) anomaly detection.
Your task is to analyze log sequences and make binary classifications: Anomaly or Normal.

HDFS anomaly indicators include:
- Replication failures or under-replicated blocks
- Checksum / data integrity errors
- Datanode communication failures or timeouts
- Blocks being deleted or lost unexpectedly
- Error or Exception log entries

Normal HDFS operations include: block allocation, receiving blocks, addStoredBlock updates, pipeline setup.

Respond ONLY with:
Label: [Anomaly/Normal]
Reason: [brief explanation]"""

    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"  [{i+1}/{len(sample)}] Block {seq.get('block_id')} ({label})", end=" ", flush=True)
        try:
            prompt = agent._build_hdfs_anomaly_prompt(seq)
            response = agent._call_llm(prompt, zero_shot_system)
            result = agent._parse_hdfs_anomaly_response(response, seq)
            pred = result.get("predicted_anomaly", False)
            predictions.append({"block_id": seq.get("block_id"), "predicted_anomaly": pred,
                                 "confidence": result.get("confidence", 0)})
            ground_truth.append({"block_id": seq.get("block_id"), "actual_anomaly": seq.get("is_anomaly", False)})
            print("✓" if pred == seq.get("is_anomaly") else "✗")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    elapsed = time.time() - t0
    metrics = calc_metrics(predictions, ground_truth)
    metrics.update(elapsed_sec=round(elapsed, 1), errors=errors, model=model,
                   provider=provider, mode="zero-shot")

    _print_metrics(metrics, elapsed, errors)
    return metrics, predictions, ground_truth


def run_model_few_shot(provider, model, sample, demo_pool):
    """Few-shot ICL run — builds demonstrations per query from demo_pool."""
    print(f"\n{'='*60}")
    print(f"[FEW-SHOT k={FEW_SHOT_K}] Model: {model}  (provider: {provider})")
    print(f"{'='*60}")

    agent = LLMAgent({"provider": provider, "model": model})
    # fixed demo set for the whole run (consistent comparison)
    demos = build_few_shot_demonstrations(demo_pool, k=FEW_SHOT_K)
    print(f"  Demonstrations: {sum(1 for d in demos if d.get('is_anomaly'))} anomalous, "
          f"{sum(1 for d in demos if not d.get('is_anomaly'))} normal")

    system_prompt = """You are an expert in HDFS (Hadoop Distributed File System) anomaly detection.
Your task is to analyze log sequences and make binary classifications: Anomaly or Normal.

HDFS anomaly indicators include:
- Replication failures or under-replicated blocks
- Checksum / data integrity errors
- Datanode communication failures or timeouts
- Blocks being deleted or lost unexpectedly
- Error or Exception log entries

Normal HDFS operations include: block allocation, receiving blocks, addStoredBlock updates, pipeline setup.

Use the labeled examples provided to calibrate your judgment. Respond ONLY with:
Label: [Anomaly/Normal]
Reason: [brief explanation]"""

    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"  [{i+1}/{len(sample)}] Block {seq.get('block_id')} ({label})", end=" ", flush=True)
        try:
            prompt = build_icl_prompt(seq, demos)
            response = agent._call_llm(prompt, system_prompt)
            result = agent._parse_hdfs_anomaly_response(response, seq)
            pred = result.get("predicted_anomaly", False)
            predictions.append({"block_id": seq.get("block_id"), "predicted_anomaly": pred,
                                 "confidence": result.get("confidence", 0)})
            ground_truth.append({"block_id": seq.get("block_id"), "actual_anomaly": seq.get("is_anomaly", False)})
            print("✓" if pred == seq.get("is_anomaly") else "✗")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    elapsed = time.time() - t0
    metrics = calc_metrics(predictions, ground_truth)
    metrics.update(elapsed_sec=round(elapsed, 1), errors=errors, model=model,
                   provider=provider, mode=f"few-shot-k{FEW_SHOT_K}")

    _print_metrics(metrics, elapsed, errors)
    return metrics, predictions, ground_truth


def _print_metrics(metrics, elapsed, errors):
    print(f"\n  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  "
          f"F1: {metrics['f1']:.3f}  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  Time: {elapsed:.1f}s  Errors: {errors}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sample, demo_pool = load_sample()
    all_results = {}

    for provider, model in MODELS:
        # --- zero-shot (original behaviour) ---
        zs_metrics, zs_preds, zs_truth = run_model_zero_shot(provider, model, sample)
        # --- few-shot ICL (new) ---
        fs_metrics, fs_preds, fs_truth = run_model_few_shot(provider, model, sample, demo_pool)

        all_results[model] = {
            "zero_shot": {"metrics": zs_metrics, "predictions": zs_preds, "ground_truth": zs_truth},
            "few_shot":  {"metrics": fs_metrics, "predictions": fs_preds, "ground_truth": fs_truth},
        }

    # Summary comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY  (zero-shot vs few-shot ICL)")
    print(f"{'='*70}")
    print(f"{'Model + Mode':<45} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Accuracy':>9}")
    print("-" * 80)
    for model, data in all_results.items():
        for mode_key in ("zero_shot", "few_shot"):
            m = data[mode_key]["metrics"]
            tag = f"{model} [{m['mode']}]"
            print(f"{tag:<45} {m['precision']:>9.3f} {m['recall']:>7.3f} "
                  f"{m['f1']:>7.3f} {m['accuracy']:>9.3f}")
        print()

    # Delta summary
    print(f"{'='*70}")
    print("ICL DELTA  (few-shot minus zero-shot)")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'ΔPrecision':>10} {'ΔRecall':>8} {'ΔF1':>8} {'ΔAccuracy':>10}")
    print("-" * 75)
    for model, data in all_results.items():
        zs = data["zero_shot"]["metrics"]
        fs = data["few_shot"]["metrics"]
        print(f"{model:<35} "
              f"{fs['precision']-zs['precision']:>+10.3f} "
              f"{fs['recall']-zs['recall']:>+8.3f} "
              f"{fs['f1']-zs['f1']:>+8.3f} "
              f"{fs['accuracy']-zs['accuracy']:>+10.3f}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/model_benchmark_icl_{ts}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
