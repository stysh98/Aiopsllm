#!/usr/bin/env python3
"""
Benchmark script: compare llama-3.1-8b-instant vs llama-3.3-70b-versatile
on the HDFS anomaly detection task using the same sample.
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
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "llama-3.3-70b-versatile"),
]

SAMPLE_SIZE = 30  # keep small to limit API cost; increase for more signal
MAX_NORMAL = 15


def load_sample():
    lab = AIOpsLab()
    dataset = lab.load_dataset("hdfs")
    sequences = dataset.get("sequences", [])
    normal = [s for s in sequences if not s.get("is_anomaly")]
    anomalous = [s for s in sequences if s.get("is_anomaly")]
    sample = anomalous[:SAMPLE_SIZE // 2] + normal[:MAX_NORMAL]
    print(f"Sample: {len(sample)} sequences ({sum(1 for s in sample if s.get('is_anomaly'))} anomalous, "
          f"{sum(1 for s in sample if not s.get('is_anomaly'))} normal)")
    return sample


def calc_metrics(predictions, ground_truth):
    tp = fp = tn = fn = 0
    for p, t in zip(predictions, ground_truth):
        pa, ta = p["predicted_anomaly"], t["actual_anomaly"]
        if pa and ta:     tp += 1
        elif pa and not ta: fp += 1
        elif not pa and not ta: tn += 1
        else:             fn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=prec, recall=rec, f1=f1, accuracy=acc)


def run_model(provider, model, sample):
    print(f"\n{'='*60}")
    print(f"Model: {model}  (provider: {provider})")
    print(f"{'='*60}")

    agent = LLMAgent({"provider": provider, "model": model})
    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"  [{i+1}/{len(sample)}] Block {seq.get('block_id')} ({label})", end=" ", flush=True)
        try:
            result = agent.analyze_hdfs_anomaly(seq)
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
    metrics["elapsed_sec"] = round(elapsed, 1)
    metrics["errors"] = errors
    metrics["model"] = model
    metrics["provider"] = provider

    print(f"\n  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  "
          f"F1: {metrics['f1']:.3f}  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  Time: {elapsed:.1f}s  Errors: {errors}")
    return metrics, predictions, ground_truth


def main():
    sample = load_sample()
    all_results = {}

    for provider, model in MODELS:
        metrics, preds, truth = run_model(provider, model, sample)
        all_results[model] = {"metrics": metrics, "predictions": preds, "ground_truth": truth}

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Accuracy':>9}")
    print("-" * 70)
    for model, data in all_results.items():
        m = data["metrics"]
        print(f"{model:<35} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['accuracy']:>9.3f}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/model_benchmark_{ts}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
