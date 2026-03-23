# AIOpsLab: LLM-Enhanced Anomaly Detection Framework

A research framework for evaluating Large Language Models (LLMs) in AIOps scenarios, specifically for HDFS log anomaly detection and root cause analysis.

## Overview

This project demonstrates the application of LLMs for automated anomaly detection in distributed systems, using HDFS logs as the primary dataset. The framework evaluates LLM performance on both balanced and realistic (imbalanced) datasets to understand real-world applicability.

## Key Features

- **LLM-based Anomaly Detection**: Uses Groq's Llama 3.1 8B model for log analysis
- **Comprehensive Evaluation**: Tests on both balanced (50% anomalies) and realistic (2.9% anomalies) datasets
- **Root Cause Analysis**: Provides detailed RCA with remediation suggestions
- **Performance Metrics**: Calculates precision, recall, F1-score, and accuracy
- **Containerized Environment**: Docker-based setup for reproducibility

## Architecture

```
AIOpsLab Framework
├── Core Framework (aiopslab/)     # Main framework logic
├── LLM Agent                      # Groq/OpenAI integration
├── Dataset Loaders               # HDFS and RCAEval data processing
├── Experiment Engine             # Configurable experiment execution
└── Evaluation Metrics            # Performance analysis tools
```

## Quick Start

1. **Setup Environment**
   ```bash
   # Start Docker Desktop
   open -a Docker
   
   # Configure API keys
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Run the Framework**
   ```bash
   make build && make up    # Build and start container
   make run-hdfs           # Run HDFS anomaly detection
   ```

3. **View Results**
   ```bash
   python3 analyze_hdfs_metrics.py     # Detailed analysis
   python3 classification_metrics_table.py  # Performance metrics
   ```

## Project Structure

```
aiopslab/
├── core/
│   ├── framework.py        # Main framework orchestration
│   └── experiment.py       # Experiment execution logic
├── agents/
│   └── llm_agent.py       # LLM integration (Groq/OpenAI)
├── datasets/
│   ├── adapter.py         # Dataset interface
│   └── loaders/           # HDFS and RCAEval loaders
experiments/
├── hdfs_anomaly.yaml      # HDFS anomaly detection config
└── demo_for_professor.yaml # Demo experiment
results/
├── stage4_evaluation_results_*.json  # Evaluation results
└── hdfs_rcaeval_integration_*.json   # Integration results
```

## Key Results

### Performance on Balanced Dataset (50% anomalies):
- **Accuracy**: 68%
- **Precision**: 71.4%
- **Recall**: 60%
- **F1-Score**: 65.2%

### Performance on Realistic Dataset (2.9% anomalies):
- **Accuracy**: 78.4%
- **Precision**: 8.1% ⚠️
- **Recall**: 62.1%
- **F1-Score**: 14.3% ⚠️

### Key Findings:
1. **Class Imbalance Challenge**: Performance degrades significantly on realistic data
2. **High False Positive Rate**: 92% of anomaly alerts would be false alarms
3. **Production Readiness**: Requires improvement for real-world deployment

## Research Contributions

1. **LLM Evaluation Framework**: Systematic evaluation of LLMs for log anomaly detection
2. **Realistic vs Balanced Testing**: Demonstrates importance of testing on realistic data distributions
3. **Comprehensive Metrics**: Beyond accuracy - focuses on precision/recall trade-offs
4. **Actionable Insights**: Provides specific recommendations for improvement

## Configuration

The framework uses environment variables for LLM configuration:

```bash
# Primary LLM (Groq)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key

# Backup LLM (OpenAI)
OPENAI_API_KEY=your_openai_key
```

## Usage Examples

### Run Complete Evaluation
```bash
make run-hdfs  # Runs anomaly detection with evaluation
```

### Analyze Results
```bash
python3 analyze_hdfs_metrics.py  # Detailed misclassification analysis
```

### Custom Experiments
```bash
# Edit experiments/hdfs_anomaly.yaml for custom configurations
docker exec -it aiopslab-framework aiopslab run /aiopslab/experiments/hdfs_anomaly.yaml
```

## Future Work

1. **Improve Few-shot Examples**: Better prompt engineering for realistic data
2. **Ensemble Methods**: Combine multiple LLM approaches
3. **Active Learning**: Iterative improvement with feedback
4. **Multi-modal Analysis**: Incorporate metrics alongside logs

## Academic Context

This framework was developed for research in applying Large Language Models to AIOps challenges, specifically focusing on the gap between laboratory performance and real-world applicability in anomaly detection systems.
