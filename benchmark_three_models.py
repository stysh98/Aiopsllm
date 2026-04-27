#!/usr/bin/env python3
"""
Benchmark: gpt-oss-120b, qwen3-32b, llama-4-scout
on HDFS anomaly detection with the balanced system prompt.
"""
import os
import sys
import json
import time
from datetime import datetime

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
    ("groq", "openai/gpt-oss-120b",                       "GPT-OSS 120B"),
    ("groq", "qwen/qwen3-32b",                             "Qwen3 32B"),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct",  "Llama 4 Scout 17B"),
]

SAMPLE_SIZE = 30
MAX_NORMAL  = 15

# Balanced system prompt — no Normal bias
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


def load_sample():
    lab = AIOpsLab()
    dataset = lab.load_dataset("hdfs")
    sequences = dataset.get("sequences", [])
    normal    = [s for s in sequences if not s.get("is_anomaly")]
    anomalous = [s for s in sequences if s.get("is_anomaly")]
    sample = anomalous[:SAMPLE_SIZE // 2] + normal[:MAX_NORMAL]
    print(f"Sample: {len(sample)} sequences "
          f"({sum(1 for s in sample if s.get('is_anomaly'))} anomalous, "
          f"{sum(1 for s in sample if not s.get('is_anomaly'))} normal)")
    return sample


def calc_metrics(predictions, ground_truth):
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


def run_model(provider, model_id, display_name, sample):
    print(f"\n{'='*65}")
    print(f"Model: {display_name}  ({model_id})")
    print(f"{'='*65}")

    agent = LLMAgent({"provider": provider, "model": model_id})
    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"  [{i+1:2d}/{len(sample)}] Block {seq.get('block_id')} ({label})", end=" ", flush=True)
        try:
            prompt   = agent._build_hdfs_anomaly_prompt(seq)
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
            correct = pred == seq.get("is_anomaly")
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
    )

    print(f"\n  Precision: {metrics['precision']:.3f}  "
          f"Recall: {metrics['recall']:.3f}  "
          f"F1: {metrics['f1']:.3f}  "
          f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} "
          f"TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  Time: {elapsed:.1f}s  Errors: {errors}")
    return metrics, predictions, ground_truth


def main():
    sample = load_sample()
    all_results = {}

    for provider, model_id, display_name in MODELS:
        metrics, preds, truth = run_model(provider, model_id, display_name, sample)
        all_results[model_id] = {
            "display_name": display_name,
            "metrics": metrics,
            "predictions": preds,
            "ground_truth": truth,
        }

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY  (balanced prompt)")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Accuracy':>9} {'Time(s)':>8}")
    print("-" * 85)
    for model_id, data in all_results.items():
        m = data["metrics"]
        print(f"{data['display_name']:<40} "
              f"{m['precision']:>9.3f} "
              f"{m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} "
              f"{m['accuracy']:>9.3f} "
              f"{m['elapsed_sec']:>8.1f}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/three_model_benchmark_{ts}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out_path, all_results


if __name__ == "__main__":
    main()
