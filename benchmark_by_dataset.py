#!/usr/bin/env python3
"""
Benchmark: gpt-oss-120b, qwen3-32b, llama-4-scout
Run on BOTH datasets (HDFS + RCAEval) and report results separately per dataset.
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

# ── HDFS config ────────────────────────────────────────────────────────────
HDFS_SAMPLE_SIZE = 30
HDFS_MAX_NORMAL  = 15

HDFS_SYSTEM_PROMPT = """You are an expert in HDFS (Hadoop Distributed File System) anomaly detection.
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

# ── RCAEval config ─────────────────────────────────────────────────────────
RCAEVAL_SAMPLE_SIZE = 20   # RCA is slower/heavier; keep manageable

RCAEVAL_SYSTEM_PROMPT = """You are an expert in microservice system root cause analysis.
Analyze the provided metrics, logs, and traces to identify the root cause of the failure.

Focus on:
- Which service is the root cause (not just a downstream victim)
- What type of fault it is (cpu/mem/disk/delay/loss)
- Evidence from metrics, logs, and traces that supports your conclusion

Respond ONLY with:
Root Cause Service: [service name]
Fault Type: [cpu/mem/disk/delay/loss/other]
Confidence: [0-100]%
Reason: [brief explanation]"""


# ══════════════════════════════════════════════════════════════════════════
# HDFS helpers
# ══════════════════════════════════════════════════════════════════════════

def load_hdfs_sample():
    lab = AIOpsLab()
    dataset = lab.load_dataset("hdfs")
    sequences = dataset.get("sequences", [])
    normal    = [s for s in sequences if not s.get("is_anomaly")]
    anomalous = [s for s in sequences if s.get("is_anomaly")]
    sample = anomalous[:HDFS_SAMPLE_SIZE // 2] + normal[:HDFS_MAX_NORMAL]
    print(f"  HDFS sample: {len(sample)} sequences "
          f"({sum(1 for s in sample if s.get('is_anomaly'))} anomalous, "
          f"{sum(1 for s in sample if not s.get('is_anomaly'))} normal)")
    return sample


def calc_classification_metrics(predictions, ground_truth):
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


def run_hdfs(provider, model_id, display_name, sample):
    print(f"\n  ── {display_name} on HDFS ──")
    agent = LLMAgent({"provider": provider, "model": model_id})
    predictions, ground_truth = [], []
    errors = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        label = "Anomalous" if seq.get("is_anomaly") else "Normal"
        print(f"    [{i+1:2d}/{len(sample)}] {seq.get('block_id')} ({label})", end=" ", flush=True)
        try:
            prompt   = agent._build_hdfs_anomaly_prompt(seq)
            response = agent._call_llm(prompt, HDFS_SYSTEM_PROMPT)
            result   = agent._parse_hdfs_anomaly_response(response, seq)
            pred     = result.get("predicted_anomaly", False)
            predictions.append({"block_id": seq.get("block_id"), "predicted_anomaly": pred})
            ground_truth.append({"block_id": seq.get("block_id"), "actual_anomaly": seq.get("is_anomaly", False)})
            print("✓" if pred == seq.get("is_anomaly") else "✗")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    elapsed = time.time() - t0
    metrics = calc_classification_metrics(predictions, ground_truth)
    metrics.update(elapsed_sec=round(elapsed, 1), errors=errors,
                   model=model_id, display_name=display_name, provider=provider)

    print(f"    → Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  Accuracy={metrics['accuracy']:.3f}  "
          f"({elapsed:.0f}s, {errors} errors)")
    return metrics, predictions, ground_truth


# ══════════════════════════════════════════════════════════════════════════
# RCAEval helpers
# ══════════════════════════════════════════════════════════════════════════

def load_rcaeval_sample():
    lab = AIOpsLab()
    dataset = lab.load_dataset("rcaeval")
    sequences = dataset.get("sequences", [])
    if not sequences:
        print("  ⚠  No RCAEval sequences found — skipping RCAEval benchmark")
        return []
    # Sample evenly across fault types
    from collections import defaultdict
    by_fault = defaultdict(list)
    for s in sequences:
        by_fault[s.get("fault_type", "unknown")].append(s)
    sample = []
    per_fault = max(1, RCAEVAL_SAMPLE_SIZE // max(len(by_fault), 1))
    for ft, cases in sorted(by_fault.items()):
        sample.extend(cases[:per_fault])
    sample = sample[:RCAEVAL_SAMPLE_SIZE]
    print(f"  RCAEval sample: {len(sample)} cases across "
          f"{len(set(s.get('fault_type') for s in sample))} fault types, "
          f"{len(set(s.get('service') for s in sample))} services")
    return sample


def _build_rcaeval_prompt(seq):
    """Build a concise prompt from a RCAEval sequence dict."""
    lines = [
        f"Case: {seq.get('case_id', 'unknown')}",
        f"System: {seq.get('system', 'unknown')}",
        f"Fault Injection Time: {seq.get('inject_time', 'unknown')}",
        "",
    ]

    # Metrics summary
    metrics = seq.get("metrics_data", {})
    if metrics:
        lines.append("METRICS (avg over window):")
        time_col = metrics.get("time", [])
        for col, vals in metrics.items():
            if col == "time" or not vals:
                continue
            try:
                avg = sum(vals) / len(vals)
                mx  = max(vals)
                lines.append(f"  {col}: avg={avg:.2f}, max={mx:.2f}")
            except Exception:
                pass
        lines.append("")

    # Logs (first 15)
    logs = seq.get("logs_data")
    if logs is not None:
        try:
            log_rows = logs.head(15).to_dict("records") if hasattr(logs, "head") else logs[:15]
            lines.append(f"LOGS (first {min(15, len(log_rows))} of {len(logs)}):")
            for row in log_rows:
                msg = row.get("message", row.get("content", str(row)))
                lines.append(f"  {msg[:120]}")
            lines.append("")
        except Exception:
            pass

    # Traces (first 10)
    traces = seq.get("traces_data")
    if traces is not None:
        try:
            trace_rows = traces.head(10).to_dict("records") if hasattr(traces, "head") else traces[:10]
            lines.append(f"TRACES (first {min(10, len(trace_rows))}):")
            for row in trace_rows:
                svc  = row.get("service", row.get("serviceName", "?"))
                dur  = row.get("duration", row.get("latency", "?"))
                stat = row.get("status", row.get("statusCode", "?"))
                lines.append(f"  service={svc}  duration={dur}  status={stat}")
            lines.append("")
        except Exception:
            pass

    lines.append("Based on the evidence above, identify the root cause service and fault type.")
    return "\n".join(lines)


def _parse_rcaeval_response(response, seq):
    """Extract predicted service and fault type from model response."""
    resp_lower = response.lower()
    predicted_service = "unknown"
    predicted_fault   = "unknown"
    confidence        = 0.5

    for line in response.split("\n"):
        ll = line.lower()
        if "root cause service:" in ll:
            predicted_service = line.split(":", 1)[-1].strip().lower()
        elif "fault type:" in ll:
            predicted_fault = line.split(":", 1)[-1].strip().lower()
        elif "confidence:" in ll:
            try:
                conf_str = line.split(":", 1)[-1].strip().replace("%", "").strip()
                confidence = float(conf_str) / 100.0
            except Exception:
                confidence = 0.5

    actual_service = seq.get("root_cause_service", seq.get("service", "unknown"))
    actual_fault   = seq.get("fault_type", "unknown")

    # Flexible match: predicted contains actual or vice versa
    service_correct = (
        actual_service.lower() in predicted_service or
        predicted_service in actual_service.lower()
    )
    fault_correct = (
        actual_fault.lower() in predicted_fault or
        predicted_fault in actual_fault.lower()
    )

    return {
        "case_id":            seq.get("case_id"),
        "predicted_service":  predicted_service,
        "actual_service":     actual_service,
        "predicted_fault":    predicted_fault,
        "actual_fault":       actual_fault,
        "service_correct":    service_correct,
        "fault_correct":      fault_correct,
        "both_correct":       service_correct and fault_correct,
        "confidence":         confidence,
        "raw_response":       response,
    }


def calc_rca_metrics(results):
    n = len(results)
    if n == 0:
        return {}
    service_acc = sum(1 for r in results if r["service_correct"]) / n
    fault_acc   = sum(1 for r in results if r["fault_correct"])   / n
    both_acc    = sum(1 for r in results if r["both_correct"])     / n
    avg_conf    = sum(r["confidence"] for r in results) / n
    return dict(
        total=n,
        service_accuracy=round(service_acc, 3),
        fault_accuracy=round(fault_acc, 3),
        both_accuracy=round(both_acc, 3),
        avg_confidence=round(avg_conf, 3),
    )


def run_rcaeval(provider, model_id, display_name, sample):
    print(f"\n  ── {display_name} on RCAEval ──")
    if not sample:
        print("    (skipped — no data)")
        return {}, []

    agent = LLMAgent({"provider": provider, "model": model_id})
    results = []
    errors  = 0
    t0 = time.time()

    for i, seq in enumerate(sample):
        case_id = seq.get("case_id", f"case_{i}")
        actual  = f"{seq.get('service','?')}_{seq.get('fault_type','?')}"
        print(f"    [{i+1:2d}/{len(sample)}] {case_id} (actual: {actual})", end=" ", flush=True)
        try:
            prompt   = _build_rcaeval_prompt(seq)
            response = agent._call_llm(prompt, RCAEVAL_SYSTEM_PROMPT)
            result   = _parse_rcaeval_response(response, seq)
            results.append(result)
            mark = "✓" if result["both_correct"] else ("~" if result["service_correct"] or result["fault_correct"] else "✗")
            print(f"{mark}  pred={result['predicted_service']}/{result['predicted_fault']}")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    elapsed = time.time() - t0
    metrics = calc_rca_metrics(results)
    metrics.update(elapsed_sec=round(elapsed, 1), errors=errors,
                   model=model_id, display_name=display_name, provider=provider)

    print(f"    → Service Acc={metrics.get('service_accuracy', 0):.3f}  "
          f"Fault Acc={metrics.get('fault_accuracy', 0):.3f}  "
          f"Both Acc={metrics.get('both_accuracy', 0):.3f}  "
          f"({elapsed:.0f}s, {errors} errors)")
    return metrics, results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def print_hdfs_table(hdfs_results):
    print(f"\n{'═'*85}")
    print("DATASET: HDFS  —  Anomaly Detection")
    print(f"{'═'*85}")
    print(f"{'Model':<40} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Accuracy':>9} {'Time(s)':>8}")
    print("─" * 85)
    for model_id, data in hdfs_results.items():
        m = data["metrics"]
        print(f"{data['display_name']:<40} "
              f"{m['precision']:>9.3f} "
              f"{m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} "
              f"{m['accuracy']:>9.3f} "
              f"{m['elapsed_sec']:>8.1f}")
        print(f"  {'':40} TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


def print_rcaeval_table(rcaeval_results):
    print(f"\n{'═'*85}")
    print("DATASET: RCAEval  —  Root Cause Analysis")
    print(f"{'═'*85}")
    print(f"{'Model':<40} {'Svc Acc':>8} {'Fault Acc':>10} {'Both Acc':>9} {'Avg Conf':>9} {'Time(s)':>8}")
    print("─" * 85)
    for model_id, data in rcaeval_results.items():
        m = data["metrics"]
        if not m:
            print(f"{data['display_name']:<40}  (skipped — no RCAEval data)")
            continue
        print(f"{data['display_name']:<40} "
              f"{m.get('service_accuracy', 0):>8.3f} "
              f"{m.get('fault_accuracy', 0):>10.3f} "
              f"{m.get('both_accuracy', 0):>9.3f} "
              f"{m.get('avg_confidence', 0):>9.3f} "
              f"{m.get('elapsed_sec', 0):>8.1f}")


def main():
    print("Loading datasets...")
    print()

    # ── Load samples ──────────────────────────────────────────────────────
    print("HDFS:")
    hdfs_sample = load_hdfs_sample()

    print("\nRCAEval:")
    rcaeval_sample = load_rcaeval_sample()

    hdfs_results    = {}
    rcaeval_results = {}

    # ── Run each model on both datasets ───────────────────────────────────
    for provider, model_id, display_name in MODELS:
        print(f"\n{'='*65}")
        print(f"MODEL: {display_name}  ({model_id})")
        print(f"{'='*65}")

        # HDFS
        h_metrics, h_preds, h_truth = run_hdfs(provider, model_id, display_name, hdfs_sample)
        hdfs_results[model_id] = {
            "display_name": display_name,
            "metrics": h_metrics,
            "predictions": h_preds,
            "ground_truth": h_truth,
        }

        # RCAEval
        r_metrics, r_results = run_rcaeval(provider, model_id, display_name, rcaeval_sample)
        rcaeval_results[model_id] = {
            "display_name": display_name,
            "metrics": r_metrics,
            "results": [{k: v for k, v in r.items() if k != "raw_response"} for r in r_results],
        }

    # ── Print summary tables ──────────────────────────────────────────────
    print_hdfs_table(hdfs_results)
    print_rcaeval_table(rcaeval_results)

    # ── Save results ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/by_dataset_benchmark_{ts}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"hdfs": hdfs_results, "rcaeval": rcaeval_results}, f, indent=2)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
