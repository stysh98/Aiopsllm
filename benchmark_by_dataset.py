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

RCAEVAL_SYSTEM_PROMPT = """You are an expert in microservice root cause analysis.
A fault has been injected into one service in the system. Your job is to identify WHICH service
and WHAT type of fault it is, using the metric anomaly signals provided.

Key principle: the root cause service will show a dramatic spike in ONE metric type
(cpu/mem/disk/latency/error-rate) right after the fault injection time. Other services
may show mild downstream effects but will NOT have the same magnitude spike.

Fault type definitions:
- cpu:   CPU usage spikes to near 100% on the root cause service
- mem:   Memory usage grows abnormally on the root cause service
- disk:  Disk I/O or disk usage spikes on the root cause service
- delay: Latency/response-time spikes sharply on the root cause service
- loss:  Error rate or packet loss spikes on the root cause service

Respond ONLY with these exact lines:
Root Cause Service: [service name, e.g. adservice]
Fault Type: [cpu/mem/disk/delay/loss]
Confidence: [0-100]%
Reason: [one sentence citing the specific metric and its before/after values]"""


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

    from collections import defaultdict

    # Prefer Online Boutique (OB) cases — they have 12 services (clean signal)
    # vs Train Ticket (60+ services) or Sock Shop.
    # Within OB, prefer RE1-OB (metrics only, very clean) then RE2-OB (has logs+traces).
    # Use instance=1 only to keep one representative per scenario.
    system_priority = {"online-boutique": 0, "sock-shop": 1, "train-ticket": 2}
    bench_priority  = {"re1": 0, "re2": 1, "re3": 2}

    sequences_with_metrics = [
        s for s in sequences
        if s.get("has_metrics") and s.get("instance", 1) == 1
    ]
    sequences_with_metrics.sort(key=lambda s: (
        system_priority.get(s.get("system", "train-ticket"), 2),
        bench_priority.get(s.get("benchmark", "re3"), 2),
    ))

    # Sample evenly across the 5 core fault types
    core_faults = ["cpu", "mem", "disk", "delay", "loss"]
    by_fault = defaultdict(list)
    for s in sequences_with_metrics:
        ft = s.get("fault_type", "unknown")
        if ft in core_faults:
            by_fault[ft].append(s)

    # Within each fault type, pick different services (not 4 instances of the same one)
    per_fault = max(1, RCAEVAL_SAMPLE_SIZE // len(core_faults))
    sample = []
    for ft in core_faults:
        cases = by_fault[ft]
        # Pick one case per unique service, up to per_fault
        seen_services = set()
        picked = []
        for c in cases:
            svc = c.get("service", "unknown")
            if svc not in seen_services:
                seen_services.add(svc)
                picked.append(c)
            if len(picked) >= per_fault:
                break
        sample.extend(picked)
    sample = sample[:RCAEVAL_SAMPLE_SIZE]

    print(f"  RCAEval sample: {len(sample)} cases across "
          f"{len(set(s.get('fault_type') for s in sample))} fault types, "
          f"{len(set(s.get('service') for s in sample))} services "
          f"({', '.join(sorted(set(s.get('system','?') for s in sample)))})")
    return sample


def _build_rcaeval_prompt(seq):
    """
    Build a diagnostic prompt using before/after metric deltas around the
    fault injection time.  This is the key improvement over the old version
    which only showed raw averages — the delta immediately highlights the
    anomalous service.
    """
    import numpy as np

    inject_time = seq.get("inject_time", 0)
    case_id     = seq.get("case_id", "unknown")
    system      = seq.get("system", "unknown")
    lines       = []

    lines.append(f"Case: {case_id}")
    lines.append(f"System: {system}")
    lines.append(f"Fault Injection Timestamp: {inject_time}")
    lines.append("")

    # ── Metric delta analysis ─────────────────────────────────────────────
    metrics = seq.get("metrics_data", {})
    if metrics and inject_time:
        time_vals = metrics.get("time", [])

        # Build parallel arrays: time → index
        before_idx = [i for i, t in enumerate(time_vals) if t < inject_time]
        after_idx  = [i for i, t in enumerate(time_vals) if t >= inject_time]

        if before_idx and after_idx:
            # Extract service names from metric column names
            # Columns look like: adservice_cpu, adservice_mem, adservice_latency …
            all_cols = [c for c in metrics if c != "time" and metrics[c]]
            services = sorted(set(
                "_".join(c.split("_")[:-1])
                for c in all_cols
                if "_" in c
            ))

            lines.append(f"SERVICES IN THIS SYSTEM: {', '.join(services)}")
            lines.append("")

            # Compute per-metric before/after means and ratio
            metric_deltas = []
            for col in all_cols:
                vals = metrics[col]
                try:
                    b_mean = float(np.mean([vals[i] for i in before_idx]))
                    a_mean = float(np.mean([vals[i] for i in after_idx]))
                    a_max  = float(np.max([vals[i] for i in after_idx]))
                    # ratio: how many times larger is after vs before?
                    ratio  = (a_mean / b_mean) if b_mean > 0.01 else (a_mean if a_mean > 0 else 0)
                    metric_deltas.append((col, b_mean, a_mean, a_max, ratio))
                except Exception:
                    pass

            # Sort by ratio descending — most anomalous first
            metric_deltas.sort(key=lambda x: x[4], reverse=True)

            lines.append("METRIC ANOMALY RANKING (sorted by after/before ratio, highest = most anomalous):")
            lines.append(f"  {'Metric':<45} {'Before':>10} {'After(avg)':>12} {'After(max)':>12} {'Ratio':>8}")
            lines.append("  " + "-" * 93)

            # Show top 15 most anomalous metrics
            for col, b, a, mx, ratio in metric_deltas[:15]:
                ratio_str = f"{ratio:.1f}x" if ratio < 1000 else ">999x"
                lines.append(f"  {col:<45} {b:>10.2f} {a:>12.2f} {mx:>12.2f} {ratio_str:>8}")

            lines.append("")

            # Per-service summary: show the single most anomalous metric per service
            lines.append("PER-SERVICE SUMMARY (worst metric per service):")
            service_worst = {}
            for col, b, a, mx, ratio in metric_deltas:
                svc = "_".join(col.split("_")[:-1]) if "_" in col else col
                if svc not in service_worst or ratio > service_worst[svc][4]:
                    service_worst[svc] = (col, b, a, mx, ratio)

            # Sort services by their worst ratio
            sorted_svcs = sorted(service_worst.items(), key=lambda x: x[1][4], reverse=True)
            for svc, (col, b, a, mx, ratio) in sorted_svcs:
                metric_type = col.split("_")[-1] if "_" in col else col
                ratio_str   = f"{ratio:.1f}x" if ratio < 1000 else ">999x"
                lines.append(f"  {svc:<35} peak metric={metric_type:<12} "
                              f"before={b:.1f}  after_avg={a:.1f}  after_max={mx:.1f}  ratio={ratio_str}")
            lines.append("")

        else:
            # Fallback: no time alignment possible, show raw averages
            lines.append("METRICS (raw averages — no injection time alignment):")
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

    # ── Logs (first 20, only if present) ─────────────────────────────────
    logs = seq.get("logs_data")
    if logs is not None:
        try:
            log_rows = logs.head(20).to_dict("records") if hasattr(logs, "head") else logs[:20]
            lines.append(f"LOGS (first {len(log_rows)}):")
            for row in log_rows:
                svc = row.get("container_name", row.get("service", ""))
                msg = row.get("message", row.get("content", str(row)))
                lvl = row.get("level", "")
                prefix = f"[{svc}][{lvl}] " if svc else ""
                lines.append(f"  {prefix}{str(msg)[:120]}")
            lines.append("")
        except Exception:
            pass

    # ── Traces (first 10, only if present) ───────────────────────────────
    traces = seq.get("traces_data")
    if traces is not None:
        try:
            trace_rows = traces.head(10).to_dict("records") if hasattr(traces, "head") else traces[:10]
            lines.append(f"TRACES (first {len(trace_rows)}):")
            for row in trace_rows:
                svc  = row.get("serviceName", row.get("service", "?"))
                dur  = row.get("duration", row.get("latency", "?"))
                stat = row.get("statusCode", row.get("status", "?"))
                op   = row.get("operationName", row.get("methodName", ""))
                lines.append(f"  service={svc:<30} op={op:<35} duration={dur}  status={stat}")
            lines.append("")
        except Exception:
            pass

    lines.append("TASK: Based on the metric anomaly ranking above, identify:")
    lines.append("  1. Which service has the largest spike after the injection time → that is the root cause service")
    lines.append("  2. Which metric type spiked (cpu/mem/disk/latency→delay/error→loss) → that is the fault type")
    return "\n".join(lines)


def _parse_rcaeval_response(response, seq):
    """Extract predicted service and fault type from model response."""
    predicted_service = "unknown"
    predicted_fault   = "unknown"
    confidence        = 0.5

    for line in response.split("\n"):
        ll = line.lower().strip()
        # Service — accept several phrasings
        if ll.startswith("root cause service:") or ll.startswith("root cause:"):
            val = line.split(":", 1)[-1].strip().lower()
            # Strip common noise like "the", quotes, brackets
            val = val.strip('"\'[]').replace("the ", "").strip()
            if val:
                predicted_service = val
        # Fault type
        elif ll.startswith("fault type:") or ll.startswith("fault:"):
            val = line.split(":", 1)[-1].strip().lower()
            val = val.strip('"\'[]').strip()
            if val:
                predicted_fault = val
        # Confidence
        elif ll.startswith("confidence:"):
            try:
                conf_str = line.split(":", 1)[-1].strip().replace("%", "").strip()
                confidence = float(conf_str) / 100.0
            except Exception:
                confidence = 0.5

    actual_service = seq.get("root_cause_service", seq.get("service", "unknown"))
    actual_fault   = seq.get("fault_type", "unknown")

    # Normalise fault type aliases the model might use
    fault_aliases = {
        "latency": "delay", "network delay": "delay", "response time": "delay",
        "packet loss": "loss", "network loss": "loss", "error rate": "loss",
        "memory": "mem", "memory leak": "mem",
        "cpu stress": "cpu", "cpu spike": "cpu",
        "disk i/o": "disk", "disk io": "disk",
    }
    for alias, canonical in fault_aliases.items():
        if alias in predicted_fault:
            predicted_fault = canonical
            break

    # Flexible match: predicted contains actual or vice versa
    service_correct = (
        actual_service.lower() in predicted_service or
        predicted_service in actual_service.lower()
    ) and predicted_service != "unknown"

    fault_correct = (
        actual_fault.lower() in predicted_fault or
        predicted_fault in actual_fault.lower()
    ) and predicted_fault != "unknown"

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
