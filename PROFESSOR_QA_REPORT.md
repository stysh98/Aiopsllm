# AIOpsLab — Professor Q&A Prep Report

> Everything you need to confidently answer questions about this project.

---

## 1. What is this project about? (The elevator pitch)

**AIOpsLab** is a research framework that evaluates Large Language Models (LLMs) on two AIOps (AI for IT Operations) tasks:

1. **HDFS Log Anomaly Detection** — given a sequence of Hadoop log lines for a storage block, classify it as *Anomaly* or *Normal*.
2. **Root Cause Analysis (RCA)** — given telemetry data (metrics, logs, traces) from a failing microservice system, identify *why* the failure happened.

The core research question is: **Can LLMs replace or augment traditional rule-based/ML anomaly detection in production systems?**

---

## 2. What datasets are used?

### HDFS Dataset (LogHub)
- **Source**: Public HDFS log dataset from LogHub
- **Size**: 11+ million log lines, grouped into ~575,000 block-level sequences
- **Labels**: Binary — Normal (0) or Anomaly (1)
- **Anomaly types**: Replication failures, checksum errors, datanode crashes, block corruption
- **How it's used**: 1,000 sequences sampled (500 anomalous + 500 normal), then 30 used for evaluation (15 anomalous + 15 normal), rest as ICL demo pool
- **Two testing modes**:
  - **Balanced** (50% anomalies) — lab/controlled setting
  - **Realistic** (2.9% anomalies) — mirrors real production distributions

### RCAEval Dataset
- **Source**: Figshare/GitHub benchmark
- **Size**: 735 failure cases across 3 microservice systems
- **Systems**: Online Boutique (Google), Sock Shop (Weaveworks), Train Ticket
- **Fault types**: CPU stress, memory leak, disk I/O, network delay, packet loss, code-level faults
- **Data types**: Time-series metrics, logs, distributed traces, service dependency graphs

---

## 3. What models were tested?

| Model | Size | Provider |
|---|---|---|
| Llama 3.1 8B Instant | 8B | Groq API |
| Llama 3.3 70B Versatile | 70B | Groq API |
| GPT-OSS 120B | 120B | Groq API |
| Llama 4 Scout 17B | 17B (MoE) | Groq API |

All models are accessed via the **Groq API** (fast inference). The framework also supports OpenAI, Anthropic, HuggingFace, and Ollama.

---

## 4. What evaluation metrics are used and why?

| Metric | Formula | Why it matters |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of all anomaly alerts, how many are real? Low precision = alert fatigue |
| **Recall** | TP / (TP + FN) | Of all real anomalies, how many are caught? Low recall = missed incidents |
| **F1-Score** | 2 × P × R / (P + R) | Harmonic mean — balances precision and recall |
| **Accuracy** | (TP+TN) / Total | Overall correctness — misleading on imbalanced data |

**Why not just accuracy?** On the realistic dataset (2.9% anomalies), a model that always predicts "Normal" gets 97.1% accuracy but catches zero anomalies. That's why F1 and precision/recall are the real measures.

---

## 5. What is In-Context Learning (ICL) and why did you study it?

**ICL (few-shot prompting)** means giving the LLM a few labeled examples inside the prompt before asking it to classify a new input — no model fine-tuning required.

**Why study it?** The zero-shot baseline had high recall (93%) but terrible precision (50%) — it was flagging almost every normal block as anomalous. The hypothesis was that showing the model examples of what "Normal" looks like would calibrate its decision threshold and reduce false positives.

### ICL strategies tested:

| Strategy | Demos | Hypothesis |
|---|---|---|
| Zero-shot (k=0) | None | Baseline |
| Few-shot k=4 balanced | 2 anomaly + 2 normal | Standard ICL |
| Few-shot k=8 balanced | 4 anomaly + 4 normal | More examples = better? |
| Few-shot k=4 normal-heavy | 1 anomaly + 3 normal | More normal examples calibrate threshold |

### ICL design principles applied (cite these if asked):
- **Balanced label distribution** (Brown et al., 2020)
- **Recency bias** — most informative demo placed last (Zhao et al., 2021)
- **No data leakage** — demos drawn from a held-out pool, not the eval set
- **Consistent format** between demos and query

---

## 6. What were the actual results? (Know these numbers)

### ICL v2 Benchmark — Llama 3.1 8B

| Condition | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 0.500 | 0.933 | **0.651** | 0.500 | 14 | 14 | 1 | 1 |
| Few-shot k=4 balanced | 0.667 | 0.267 | 0.381 | 0.567 | 4 | 2 | 13 | 11 |
| Few-shot k=8 balanced | 1.000 | 0.200 | 0.333 | 0.600 | 3 | 0 | 15 | 12 |
| **Few-shot k=4 normal-heavy** | **1.000** | **0.533** | **0.696** | **0.767** | 8 | 0 | 15 | 7 |

### ICL v2 Benchmark — Llama 3.3 70B

| Condition | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Zero-shot | 0.464 | 0.867 | 0.605 | 0.433 |
| Few-shot k=4 balanced | 0.000 | 0.000 | 0.000 | 0.500 |
| Few-shot k=8 balanced | 0.000 | 0.000 | 0.000 | 0.500 |
| Few-shot k=4 normal-heavy | 0.000 | 0.000 | 0.000 | 0.500 |

**Key takeaway**: The 8B model with normal-heavy ICL (F1=0.696) is the best configuration. The 70B model completely collapsed with any few-shot prompting.

---

## 7. What are the key findings? (The "so what")

### Finding 1: Prompt engineering matters as much as model size
The very first ICL attempt (`benchmark_models_icl.py`) had a system prompt that said *"default to Normal"* — every model predicted Normal for everything, giving F1=0.000. Removing that single instruction restored performance. **Lesson: a bad prompt can completely override model capability.**

### Finding 2: Few-shot ICL improves precision but hurts recall
Adding balanced examples (k=4, k=8) pushed precision to 100% but recall dropped to 20%. The model became too conservative. The normal-heavy strategy (1 anomaly + 3 normal demos) found the best balance: 100% precision + 53% recall = F1 of 0.696.

### Finding 3: Bigger model ≠ better performance
Llama 3.3 70B performed *worse* than 8B on this task. With few-shot prompting, 70B completely failed (predicted Normal for everything). Hypothesis: larger models may be more sensitive to prompt structure or suffer from instruction-following degradation when demonstrations conflict with their prior.

### Finding 4: Class imbalance is the critical production challenge
- Balanced dataset (50% anomalies): F1 ≈ 0.65
- Realistic dataset (2.9% anomalies): F1 ≈ 0.14, Precision ≈ 8%
- **92% of anomaly alerts would be false alarms in production**
- This is the gap between "works in the lab" and "works in production"

### Finding 5: Precision-Recall trade-off is fundamental
There is no single configuration that maximizes both. The choice depends on operational requirements:
- High recall needed → zero-shot (catch everything, accept false alarms)
- High precision needed → few-shot k=8 (no false alarms, miss most anomalies)
- Best balance → few-shot k=4 normal-heavy

---

## 8. How does the framework work? (Architecture questions)

```
User / Benchmark Script
        ↓
AIOpsLab (framework.py)          ← orchestrator
    ├── DatasetAdapter            ← loads HDFS or RCAEval data
    │       ├── HdfsLoader        ← parses 11M log lines, creates sequences
    │       └── RcaevalLoader     ← loads microservice failure cases
    └── LLMAgent (llm_agent.py)  ← builds prompts, calls API, parses responses
            ├── _build_hdfs_anomaly_prompt()
            ├── _call_llm()       ← Groq/OpenAI/Anthropic API
            └── _parse_hdfs_anomaly_response()
```

**How a single prediction works:**
1. Load HDFS log sequence for a block (e.g., `blk_4966276028193818154`)
2. Build a prompt: system prompt + (optional ICL demos) + the log lines
3. Call Groq API → get text response
4. Parse response for `Label: [Anomaly/Normal]`
5. Compare to ground truth label → TP/FP/TN/FN

---

## 9. Why HDFS logs specifically?

- **Well-studied benchmark**: HDFS v1 from LogHub is a standard dataset in log analysis research
- **Real data**: Collected from a production Hadoop cluster at IBM
- **Clear ground truth**: Block-level anomaly labels are available
- **Structured anomalies**: Failures have identifiable patterns (replication errors, checksum failures)
- **Scalable**: 11M+ log lines provide enough data for meaningful evaluation

---

## 10. What are the limitations and future work?

### Current limitations:
- **Small eval set**: Only 30 sequences per run (15 anomalous, 15 normal) — results may not generalize
- **Single dataset**: HDFS only; other log sources (Linux, Apache, etc.) not tested
- **No fine-tuning**: Only prompt engineering; fine-tuned models would likely perform better
- **API dependency**: Relies on Groq API — latency and cost are real constraints
- **70B failure unexplained**: Why 70B collapses with few-shot is not fully understood

### Future work (from the README):
1. **Ensemble methods**: Combine multiple LLM predictions
2. **Active learning**: Iterative improvement with human feedback
3. **Better few-shot selection**: Use semantic similarity to pick relevant demos
4. **Multi-modal analysis**: Combine logs + metrics + traces
5. **Fine-tuning**: Domain-specific fine-tuning on HDFS logs

---

## 11. Possible professor questions and answers

**Q: Why use LLMs for anomaly detection instead of traditional ML (e.g., Isolation Forest, LSTM)?**
> Traditional ML requires labeled training data and retraining when patterns change. LLMs bring domain knowledge from pre-training and can generalize zero-shot. The trade-off is cost, latency, and the precision/recall challenges shown in this work.

**Q: What is the difference between precision and recall, and which matters more here?**
> Precision = of all alerts raised, how many are real. Recall = of all real anomalies, how many were caught. In production, low precision causes alert fatigue (operators ignore alerts). Low recall means real incidents are missed. The right balance depends on the cost of each error — in critical systems, recall is usually prioritized.

**Q: Why did the 70B model fail completely with few-shot prompting?**
> This is an open question. One hypothesis is that the 70B model's stronger instruction-following causes it to over-weight the demonstrations and conclude that "Normal" is the safe default when it sees more normal examples. Another hypothesis is context window sensitivity — the longer prompt changes how the model attends to the query. This is a finding worth investigating further.

**Q: What is the significance of the "normal-heavy" ICL strategy?**
> It's inspired by the observation that the model's prior is biased toward "Anomaly" (zero-shot recall=93%). By showing 3 normal examples for every 1 anomaly example, we shift the model's implicit threshold. This is analogous to cost-sensitive learning in traditional ML — you weight the classes to match your operational priorities.

**Q: How do you prevent data leakage in your ICL setup?**
> The 1,000 sequences are split into an eval set (30 sequences) and a demo pool (970 sequences). ICL demonstrations are drawn exclusively from the demo pool — the eval sequences are never used as demonstrations. This ensures the model hasn't "seen" the test cases.

**Q: What does the realistic vs. balanced dataset comparison tell us?**
> It reveals the lab-to-production gap. A balanced dataset (50% anomalies) is unrealistic — in production, anomalies are rare (2.9% here). When tested on realistic data, precision drops from ~70% to ~8%, meaning 92% of alerts are false alarms. This is a critical finding for anyone considering deploying LLMs for production monitoring.

**Q: How is the RCAEval dataset used?**
> RCAEval provides 735 labeled failure cases from 3 microservice systems with multi-source telemetry (metrics, logs, traces). The LLM agent analyzes this data to identify the root cause service and fault type. The framework supports cross-analysis between HDFS anomaly detection and RCAEval RCA results.

**Q: What API/infrastructure does this use?**
> Groq API for fast LLM inference (free tier available). The framework is containerized with Docker for reproducibility. API keys are managed via `.env` file. The framework also supports OpenAI, Anthropic, and local models via Ollama.

**Q: What is the research contribution of this work?**
> 1. A systematic evaluation framework for LLMs on AIOps tasks (reproducible, multi-model, multi-dataset)
> 2. Empirical evidence that ICL can improve precision but requires careful demonstration design
> 3. Quantification of the lab-to-production performance gap due to class imbalance
> 4. The unexpected finding that larger models (70B) can underperform smaller ones (8B) on structured classification tasks

---

## 12. Quick reference — numbers to memorize

| Fact | Value |
|---|---|
| HDFS log lines | 11+ million |
| Total sequences | ~575,000 |
| Eval sample size | 30 (15 anomalous + 15 normal) |
| Demo pool size | 970 sequences |
| Best F1 achieved | **0.696** (8B, normal-heavy ICL) |
| Zero-shot 8B F1 | 0.651 |
| Zero-shot 70B F1 | 0.605 |
| Realistic dataset precision | ~8% (92% false alarm rate) |
| RCAEval cases | 735 |
| RCAEval systems | 3 (Online Boutique, Sock Shop, Train Ticket) |
| Models benchmarked | 4 (8B, 70B, 120B, Scout 17B) |
