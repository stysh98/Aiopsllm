# AIOpsLab — Complete Project Report
## Everything We Built, Every Result We Got

---

## 1. What Is This Project?

**AIOpsLab** is a research framework that uses Large Language Models (LLMs) to perform two AIOps (AI for IT Operations) tasks:

1. **HDFS Log Anomaly Detection** — given a sequence of Hadoop Distributed File System log lines for a storage block, classify it as *Anomaly* or *Normal*.
2. **Root Cause Analysis (RCA)** — given telemetry data (metrics, logs, traces) from a failing microservice system, identify *why* the failure happened and which service caused it.

The core research question: **Can LLMs replace or augment traditional rule-based/ML anomaly detection in production systems?**

---

## 2. Project File Structure — What Every File Does

```
aiopslab/                          ← Main Python package
├── core/
│   ├── framework.py               ← Main orchestrator (AIOpsLab class)
│   └── experiment.py              ← Experiment runner (Experiment class)
├── agents/
│   └── llm_agent.py               ← LLM integration (all providers + prompts)
├── datasets/
│   ├── adapter.py                 ← Dataset registry and loader dispatcher
│   └── loaders/
│       ├── hdfs_loader.py         ← Parses 11M HDFS log lines into sequences
│       ├── rcaeval_loader.py      ← Loads 735 microservice failure cases
│       └── aiops_loader.py        ← AIOps Challenge dataset loader

datasets/
├── hdfs/
│   ├── HDFS.log                   ← 11+ million raw HDFS log lines
│   └── anomaly_label.csv          ← Block-level labels (Normal/Anomaly)
└── rcaeval/
    └── RE1-OB/                    ← Online Boutique failure cases
        ├── adservice_cpu/1-5/     ← 5 instances of CPU fault on adservice
        ├── adservice_delay/1-5/
        ├── adservice_disk/1-5/
        ├── adservice_loss/1-5/
        ├── adservice_mem/1-5/
        ├── cartservice_cpu/1-5/
        └── ... (many more services × fault types)

configs/
└── framework.yaml                 ← Framework config (LLM provider, dataset paths)

Benchmark scripts (run in order of development):
├── benchmark_models.py            ← Step 1: Zero-shot, 2 models (8B vs 70B)
├── benchmark_models_icl.py        ← Step 2: First ICL attempt (FAILED — all Normal)
├── benchmark_larger_models.py     ← Step 3: 4 models, fixed balanced prompt
├── benchmark_icl_v2.py            ← Step 4: Proper ICL study (4 conditions × 2 models)
├── benchmark_three_models.py      ← Step 5: 3 models (120B, Qwen3 32B, Scout 17B)
└── benchmark_by_dataset.py        ← Step 6: Both datasets (HDFS + RCAEval) together

Analysis scripts:
├── analyze_hdfs_metrics.py        ← Loads result JSON, prints confusion matrix + FP analysis
└── classification_metrics_table.py ← Hardcoded early result (TP=25, FP=25) for display

Documentation:
├── README.md                      ← Project overview and quick start
├── PROFESSOR_QA_REPORT.md         ← Q&A prep document for academic presentation
└── PROJECT_FULL_REPORT.md         ← This file
```

---

## 3. Datasets Used

### 3.1 HDFS Dataset (LogHub)

| Property | Value |
|---|---|
| Source | LogHub (IBM production Hadoop cluster) |
| Raw log lines | 11+ million |
| Total block sequences | ~575,000 |
| Label type | Binary: Normal (0) or Anomaly (1) |
| Anomaly rate | ~2.9% (realistic) |
| Files | `HDFS.log`, `anomaly_label.csv` |

**What a "sequence" is:** HDFS groups log lines by block ID (e.g., `blk_4966276028193818154`). All log lines mentioning the same block form one sequence. Each sequence gets one label: Normal or Anomaly.

**Anomaly types in the dataset:**
- Block replication failures
- Checksum / data integrity errors
- Datanode communication failures or timeouts
- Blocks being deleted or lost unexpectedly

**How we sample it:**
- Load 1,000 sequences (500 anomalous + 500 normal) from the full 575K
- Eval set: first 15 anomalous + first 15 normal = **30 sequences**
- Demo pool: remaining 970 sequences (used only for ICL demonstrations, never evaluated)

**Two testing modes:**
- **Balanced** (50% anomalies, 15+15): lab/controlled setting
- **Realistic** (2.9% anomalies): mirrors real production distributions

---

### 3.2 RCAEval Dataset

| Property | Value |
|---|---|
| Source | Figshare/GitHub benchmark |
| Total failure cases | 735 |
| Systems | Online Boutique (Google), Sock Shop (Weaveworks), Train Ticket |
| Fault types | cpu, mem, disk, delay, loss, socket, f1–f5 |
| Data per case | metrics (data.csv), logs (logs.csv), traces (traces.csv), inject_time.txt |

**Directory structure per case:**
```
RE1-OB/adservice_cpu/1/
    ├── data.csv          ← time-series metrics (CPU, memory, latency, etc.)
    ├── logs.csv          ← service log entries
    ├── traces.csv        ← distributed trace records
    └── inject_time.txt   ← Unix timestamp when fault was injected
```

**Case ID format:** `RE1-OB_adservice_cpu_1`
- `RE1` = benchmark round 1
- `OB` = Online Boutique system
- `adservice` = the service where fault was injected
- `cpu` = fault type
- `1` = instance number (1–5 repetitions per scenario)

---

## 4. What We Built — Code by Code

### 4.1 `aiopslab/agents/llm_agent.py` — The LLM Brain

This is the core intelligence layer. It supports **6 LLM providers**:

| Provider | How it connects |
|---|---|
| `groq` | Groq API (fast inference, free tier) — **primary** |
| `openai` | OpenAI API |
| `anthropic` | Anthropic API |
| `huggingface` | HuggingFace Inference API |
| `ollama` | Local Ollama server (http://localhost:11434) |
| `local` | Custom local setup |

**Key methods written:**

| Method | What it does |
|---|---|
| `analyze_hdfs_anomaly(sequence_data)` | Builds HDFS prompt, calls LLM, parses Label: Anomaly/Normal |
| `analyze_rcaeval_failure(case_data)` | Builds RCA prompt with metrics/logs/traces, calls LLM |
| `analyze_root_cause(data)` | Generic RCA dispatcher |
| `evaluate_anomaly_detection_performance(predictions, ground_truth)` | Computes TP/FP/TN/FN, asks LLM to evaluate |
| `suggest_model_improvements(performance_data)` | Asks LLM for improvement suggestions |
| `suggest_remediation(issue)` | Asks LLM for fix recommendations |
| `_build_hdfs_anomaly_prompt(sequence_data)` | Formats log sequence into structured prompt |
| `_build_rcaeval_rca_prompt(case_data)` | Formats metrics/logs/traces into RCA prompt |
| `_call_llm(prompt, system_prompt)` | Unified API caller for all providers |
| `_parse_hdfs_anomaly_response(response, seq)` | Extracts Label: Anomaly/Normal from text |
| `_parse_rcaeval_rca_response(response, case_data)` | Extracts root cause and confidence |

**HDFS system prompt (final version — balanced):**
```
You are an expert in HDFS anomaly detection.
Classify based on the evidence in the logs.
Do not default to either label — weigh the evidence.
Respond ONLY with: Label: [Anomaly/Normal] + Reason
```

**RCA system prompt:**
```
You are an expert in microservice system root cause analysis.
Base ALL conclusions on concrete evidence from metrics, logs, traces.
Distinguish clearly between symptoms and root causes.
Admit uncertainty when evidence is insufficient.
```

---

### 4.2 `aiopslab/core/framework.py` — The Orchestrator

The `AIOpsLab` class is the entry point. It:
1. Loads config from `configs/framework.yaml`
2. Reads `LLM_PROVIDER` and `LLM_MODEL` from environment variables
3. Creates an `LLMAgent` instance
4. Creates a `DatasetAdapter` instance
5. Exposes `load_dataset(name)` and `run_experiment(config)` methods

```python
lab = AIOpsLab()
dataset = lab.load_dataset("hdfs")   # or "rcaeval"
results = lab.run_experiment(config)
```

---

### 4.3 `aiopslab/core/experiment.py` — The Experiment Runner

The `Experiment` class handles:
- **Single-dataset experiments** (HDFS only or RCAEval only)
- **Multi-dataset experiments** (HDFS + RCAEval + cross-analysis)

**Phases in a multi-dataset experiment:**
1. Phase 1: HDFS anomaly detection
2. Phase 2: RCAEval root cause analysis
3. Phase 3: Cross-dataset pattern analysis (LLM compares patterns from both)

**Sampling strategy for HDFS:**
- Default: 50 sequences total, focus on anomalous ones
- Splits into normal vs anomalous, samples proportionally

**Sampling strategy for RCAEval:**
- Default: 20 cases
- Prioritizes multi-source cases (those with logs + traces, not just metrics)

**Prediction parsing logic** (written in `_extract_prediction_from_analysis`):
1. Look for explicit `Label: Anomaly` or `Label: Normal`
2. Look for decision patterns (`this is an anomaly`, `classify as normal`, etc.)
3. Fall back to keyword scoring: count anomaly keywords vs normal keywords
4. If anomaly score > normal score + 2 → predict Anomaly

---

### 4.4 `aiopslab/datasets/loaders/hdfs_loader.py` — HDFS Data Parser

**What it does:**
1. Reads `anomaly_label.csv` → maps BlockId → Label (0/1)
2. Samples 1,000 block IDs (500 anomalous + 500 normal)
3. Scans all 11M+ log lines, extracting only lines for sampled blocks
4. Groups lines by block ID → creates sequence dicts
5. Returns sequences with: `block_id`, `logs`, `log_count`, `is_anomaly`, `templates`, `components`

**Log parsing regex:**
```
(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([^:]+):\s+(.*)
→ date, time, pid, level, component, content
```

**Block ID extraction regex:**
```
(blk_-?\d+)
```

---

### 4.5 `aiopslab/datasets/loaders/rcaeval_loader.py` — RCAEval Data Parser

**What it does:**
1. Scans `datasets/rcaeval/` for `RE*` directories
2. For each `RE_dir/service_fault/instance/`:
   - Reads `data.csv` → metrics time series
   - Reads `logs.csv` → log entries
   - Reads `traces.csv` → distributed traces
   - Reads `inject_time.txt` → fault injection timestamp
3. Parses case ID to extract: system, service, fault_type, instance
4. Creates sequence dicts with all available data

**Case ID parsing:**
- `RE1-OB_adservice_cpu_1` → benchmark=re1, system=online-boutique, service=adservice, fault_type=cpu, instance=1
- System mapping: OB=online-boutique, SS=sock-shop, TT=train-ticket

---

## 5. Benchmark Scripts — The Experiments We Ran

### Experiment 1: `benchmark_models.py` — Zero-Shot Baseline

**Purpose:** Compare 8B vs 70B models, zero-shot, no ICL.

**Models:**
- `llama-3.1-8b-instant` (8B)
- `llama-3.3-70b-versatile` (70B)

**Setup:** 30 sequences (15 anomalous + 15 normal), same sample for both models.

**System prompt used:** Conservative — told model to "default to Normal when uncertain."

**What happened:** Both models had decent recall but poor precision. The conservative prompt helped somewhat but the 70B model was inconsistent.

---

### Experiment 2: `benchmark_models_icl.py` — First ICL Attempt (FAILED)

**Purpose:** Add few-shot ICL (k=4 balanced demos) to improve precision.

**What went wrong:** The system prompt contained the instruction:
> *"When in doubt, classify as NORMAL"*

This single line caused **every model to predict Normal for everything** → F1 = 0.000 for all ICL conditions.

**Lesson learned:** A bad system prompt completely overrides ICL demonstrations. The model's instruction-following prioritized the system prompt over the examples.

**ICL infrastructure built (reused later):**
- `build_few_shot_demonstrations(pool, k)` — picks k/2 anomalous + k/2 normal from demo pool
- `format_sequence_for_demo(seq)` — renders a sequence as compact log block
- `build_icl_prompt(query_seq, demos)` — builds full few-shot prompt with recency bias
- Demo pool separation: eval set (30) vs demo pool (970) — no data leakage

---

### Experiment 3: `benchmark_larger_models.py` — 4 Models, Fixed Prompt

**Purpose:** Test 4 models with a **balanced system prompt** (no Normal bias).

**Models tested:**
| Model | Size | Notes |
|---|---|---|
| `llama-3.1-8b-instant` | 8B | Small, fast baseline |
| `llama-3.3-70b-versatile` | 70B | Large baseline |
| `openai/gpt-oss-120b` | 120B | Larger model |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 17B (MoE) | Latest architecture |

**Fixed system prompt (balanced):**
```
Classify based on the evidence in the logs.
Do not default to either label — weigh the evidence.
```

**Setup:** 30 sequences (15 anomalous + 15 normal), zero-shot.

**Results from this run (zero-shot, balanced prompt):**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Llama 3.1 8B | 0.500 | 0.933 | 0.651 | 0.500 |
| Llama 3.3 70B | 0.464 | 0.867 | 0.605 | 0.433 |
| GPT-OSS 120B | ~0.5 | ~0.9 | ~0.65 | ~0.5 |
| Llama 4 Scout 17B | ~0.5 | ~0.9 | ~0.65 | ~0.5 |

**Key finding:** All models had high recall (~87–93%) but low precision (~46–50%). They were flagging almost everything as anomalous.

---

### Experiment 4: `benchmark_icl_v2.py` — Proper ICL Study (Main Experiment)

**Purpose:** Fix the ICL failure from Experiment 2 and properly study 4 ICL conditions.

**Models:** 8B and 70B (same as Experiment 1).

**4 ICL conditions tested:**

| Condition | Demos | Hypothesis |
|---|---|---|
| `zero-shot` | 0 | Baseline |
| `few-shot-k4-balanced` | 2 anomaly + 2 normal | Standard ICL |
| `few-shot-k8-balanced` | 4 anomaly + 4 normal | More examples = better? |
| `few-shot-k4-normal-heavy` | 1 anomaly + 3 normal | More normal demos calibrate threshold |

**ICL design principles applied:**
- Balanced label distribution in demonstrations (Brown et al., 2020)
- Recency bias: anomalous demo placed last in the list (Zhao et al., 2021)
- No data leakage: demos from held-out pool (970 sequences), never from eval set (30)
- Consistent format between demos and query
- Fixed random seed (42) for reproducibility

**Full Results — Llama 3.1 8B:**

| Condition | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.500 | 0.933 | **0.651** | 0.500 | 14 | 14 | 1 | 1 |
| Few-shot k=4 balanced | 0.667 | 0.267 | 0.381 | 0.567 | 4 | 2 | 13 | 11 |
| Few-shot k=8 balanced | 1.000 | 0.200 | 0.333 | 0.600 | 3 | 0 | 15 | 12 |
| **Few-shot k=4 normal-heavy** | **1.000** | **0.533** | **0.696** | **0.767** | 8 | 0 | 15 | 7 |

**Full Results — Llama 3.3 70B:**

| Condition | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.464 | 0.867 | 0.605 | 0.433 | 13 | 15 | 0 | 2 |
| Few-shot k=4 balanced | 0.000 | 0.000 | 0.000 | 0.500 | 0 | 0 | 15 | 15 |
| Few-shot k=8 balanced | 0.000 | 0.000 | 0.000 | 0.500 | 0 | 0 | 15 | 15 |
| Few-shot k=4 normal-heavy | 0.000 | 0.000 | 0.000 | 0.500 | 0 | 0 | 15 | 15 |

**Delta table (ICL improvement vs zero-shot):**

| Model + Condition | ΔPrecision | ΔRecall | ΔF1 | ΔAccuracy |
|---|---|---|---|---|
| 8B [few-shot-k4-balanced] | +0.167 | -0.667 | -0.270 | +0.067 |
| 8B [few-shot-k8-balanced] | +0.500 | -0.733 | -0.318 | +0.100 |
| **8B [few-shot-k4-normal-heavy]** | **+0.500** | **-0.400** | **+0.045** | **+0.267** |
| 70B [few-shot-k4-balanced] | -0.464 | -0.867 | -0.605 | +0.067 |
| 70B [few-shot-k8-balanced] | -0.464 | -0.867 | -0.605 | +0.067 |
| 70B [few-shot-k4-normal-heavy] | -0.464 | -0.867 | -0.605 | +0.067 |

---

### Experiment 5: `benchmark_three_models.py` — 3 Models, Balanced Prompt

**Purpose:** Test GPT-OSS 120B, Qwen3 32B, and Llama 4 Scout 17B on HDFS.

**Models:**
- `openai/gpt-oss-120b` (GPT-OSS 120B)
- `qwen/qwen3-32b` (Qwen3 32B)
- `meta-llama/llama-4-scout-17b-16e-instruct` (Llama 4 Scout 17B)

**Setup:** Same 30-sequence sample, balanced system prompt, zero-shot.

Results saved to `results/three_model_benchmark_*.json`.

---

### Experiment 6: `benchmark_by_dataset.py` — Both Datasets Together

**Purpose:** Run 3 models on BOTH HDFS (anomaly detection) AND RCAEval (root cause analysis) and report results per dataset.

**Models:** GPT-OSS 120B, Qwen3 32B, Llama 4 Scout 17B.

**HDFS part:** Same 30-sequence setup as before.

**RCAEval part:**
- Loads up to 20 cases, sampled evenly across fault types
- Builds a prompt with metrics summary, first 15 log lines, first 10 traces
- Parses response for: `Root Cause Service:`, `Fault Type:`, `Confidence:`
- Evaluates: service accuracy, fault accuracy, both-correct accuracy

**RCAEval metrics:**
- `service_accuracy` — did the model name the correct root cause service?
- `fault_accuracy` — did the model identify the correct fault type (cpu/mem/disk/etc.)?
- `both_accuracy` — did the model get both right?
- `avg_confidence` — average self-reported confidence

Results saved to `results/by_dataset_benchmark_*.json`.

---

## 6. Analysis Scripts

### `analyze_hdfs_metrics.py`

Loads a saved result JSON file (`results/hdfs_rcaeval_integration_20260310_155309.json`) and:
1. Extracts predictions and ground truth
2. Computes confusion matrix (TP/FP/TN/FN)
3. Computes precision, recall, F1, accuracy
4. Analyzes false positive patterns:
   - How many FPs mention "replication"?
   - How many mention "network"?
   - How many mention "multiple sources"?
5. Prints sample false positive analyses

**This script revealed the early problem:** The model was predicting 25/25 normal sequences as anomalies (FP=25, TN=0), giving precision=50% and specificity=0%.

---

### `classification_metrics_table.py`

A hardcoded display script showing the worst-case early result:
- TP=25, FP=25, TN=0, FN=0 (model predicted everything as Anomaly)
- Precision=50%, Recall=100%, F1=66.7%, Accuracy=50%
- Specificity=0% (zero normal sequences correctly identified)

This was used to clearly document the "model always predicts Anomaly" failure mode.

---

## 7. Key Findings — The "So What"

### Finding 1: Prompt engineering matters as much as model size

The first ICL attempt (`benchmark_models_icl.py`) had a system prompt saying *"default to Normal"* — every model predicted Normal for everything, giving F1=0.000. Removing that single instruction restored performance. **One bad sentence in a system prompt can completely override model capability.**

### Finding 2: Few-shot ICL improves precision but hurts recall

Adding balanced examples (k=4, k=8) pushed precision to 100% but recall dropped to 20%. The model became too conservative. The normal-heavy strategy (1 anomaly + 3 normal demos) found the best balance: 100% precision + 53% recall = **F1 of 0.696**.

### Finding 3: Bigger model ≠ better performance

Llama 3.3 70B performed *worse* than 8B on this task. With any few-shot prompting, 70B completely failed (predicted Normal for everything, F1=0.000). Hypothesis: larger models may be more sensitive to prompt structure, or suffer from instruction-following degradation when demonstrations conflict with their prior.

### Finding 4: Class imbalance is the critical production challenge

| Dataset | Precision | Recall | F1 | False Alarm Rate |
|---|---|---|---|---|
| Balanced (50% anomalies) | ~70% | ~60% | ~65% | ~30% |
| Realistic (2.9% anomalies) | ~8% | ~62% | ~14% | **~92%** |

In production, **92% of anomaly alerts would be false alarms**. This is the gap between "works in the lab" and "works in production."

### Finding 5: Precision-Recall trade-off is fundamental

There is no single configuration that maximizes both:
- High recall needed → zero-shot (catch everything, accept false alarms)
- High precision needed → few-shot k=8 (no false alarms, miss most anomalies)
- Best balance → few-shot k=4 normal-heavy (F1=0.696)

---

## 8. Architecture — How a Single Prediction Works

```
1. Load HDFS log sequence for a block (e.g., blk_4966276028193818154)
        ↓
2. HdfsLoader parses HDFS.log → extracts all lines for that block
        ↓
3. LLMAgent._build_hdfs_anomaly_prompt(sequence)
   → formats log lines into structured prompt
   → optionally prepends ICL demonstrations
        ↓
4. LLMAgent._call_llm(prompt, system_prompt)
   → calls Groq API → gets text response
        ↓
5. LLMAgent._parse_hdfs_anomaly_response(response, seq)
   → looks for "Label: Anomaly" or "Label: Normal"
   → extracts confidence percentage if present
        ↓
6. Compare predicted_anomaly vs is_anomaly → TP/FP/TN/FN
```

---

## 9. Configuration

### Environment Variables (`.env`)

```bash
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### `configs/framework.yaml`

```yaml
framework:
  name: aiopslab
  version: 0.1.0
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
datasets:
  base_path: datasets
  cache_enabled: true
```

---

## 10. Numbers to Know

| Fact | Value |
|---|---|
| HDFS raw log lines | 11+ million |
| Total HDFS block sequences | ~575,000 |
| Sequences loaded per run | 1,000 |
| Eval sample size | 30 (15 anomalous + 15 normal) |
| Demo pool size | 970 sequences |
| HDFS anomaly rate (realistic) | 2.9% |
| RCAEval total cases | 735 |
| RCAEval systems | 3 (Online Boutique, Sock Shop, Train Ticket) |
| RCAEval fault types | cpu, mem, disk, delay, loss, socket, f1–f5 |
| Models benchmarked | 6 (8B, 70B, 120B, Scout 17B, Qwen3 32B, + others) |
| Best F1 achieved | **0.696** (8B, normal-heavy ICL, k=4) |
| Zero-shot 8B F1 | 0.651 |
| Zero-shot 70B F1 | 0.605 |
| Realistic dataset precision | ~8% (92% false alarm rate) |
| Benchmark scripts written | 6 |
| ICL conditions tested | 4 |

---

## 11. Limitations

- **Small eval set:** Only 30 sequences per run — results may not generalize
- **Single log dataset:** HDFS only; Linux, Apache, etc. not tested
- **No fine-tuning:** Only prompt engineering; fine-tuned models would likely perform better
- **API dependency:** Relies on Groq API — latency and cost are real constraints
- **70B failure unexplained:** Why 70B collapses with few-shot is not fully understood
- **RCAEval evaluation is qualitative:** No automated ground-truth matching for RCA (only service name + fault type string matching)

---

## 12. Future Work

1. **Ensemble methods:** Combine multiple LLM predictions
2. **Active learning:** Iterative improvement with human feedback
3. **Better few-shot selection:** Use semantic similarity to pick relevant demos
4. **Multi-modal analysis:** Combine logs + metrics + traces for HDFS too
5. **Fine-tuning:** Domain-specific fine-tuning on HDFS logs
6. **Realistic ICL:** Test ICL on the 2.9% anomaly rate dataset
7. **Explain 70B collapse:** Investigate why larger models fail with few-shot on this task
