import os
import json
import re
from typing import Dict, Any, List, Optional
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


class LLMAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = LLMProvider(config.get("provider", "groq"))
        self.model = config.get("model", "llama-3.1-8b-instant")
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.provider == LLMProvider.OPENAI:
            import openai
            return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == LLMProvider.ANTHROPIC:
            import anthropic
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == LLMProvider.GROQ:
            try:
                from groq import Groq
                return Groq(api_key=os.getenv("GROQ_API_KEY"))
            except ImportError:
                print("Groq package not installed. Install with: pip install groq")
                return None
        elif self.provider in (LLMProvider.HUGGINGFACE, LLMProvider.OLLAMA):
            return None
        return None

    def analyze_rcaeval_failure(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_rcaeval_rca_prompt(case_data)
        system_prompt = """You are an expert in microservice system root cause analysis with a focus on evidence-based reasoning.

CORE PRINCIPLES:
- Base ALL conclusions on concrete evidence from metrics, logs, traces, and service metadata
- Distinguish clearly between symptoms (what you observe) and root causes (why it happened)
- Admit uncertainty when evidence is insufficient rather than guessing
- Avoid assumptions based solely on case names or fault types

EXPERTISE AREAS:
- Microservice architecture failure patterns
- Service dependency analysis and fault propagation
- Resource bottleneck identification using metrics
- Log analysis for error patterns and timing issues
- Distributed tracing for request flow analysis
- Performance degradation root cause identification

ANALYSIS APPROACH:
1. Examine all available evidence systematically
2. Identify symptoms vs. underlying causes
3. Trace fault propagation through service dependencies
4. Provide confidence levels based on evidence quality
5. Acknowledge limitations and alternative explanations"""

        response = self._call_llm(prompt, system_prompt)
        return self._parse_rcaeval_rca_response(response, case_data)

    def analyze_hdfs_anomaly(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_hdfs_anomaly_prompt(sequence_data)
        system_prompt = """You are an expert in HDFS (Hadoop Distributed File System) anomaly detection and analysis.
Your task is to analyze log sequences and make binary classifications: Anomaly or Normal.

BE EXTREMELY CONSERVATIVE in your classifications. The default assumption should be "Normal" unless
there is CLEAR, EXPLICIT evidence of actual failures or errors.

Remember:
- Repeated operations, retries, and replication activities are typically NORMAL
- Only classify as Anomaly when you see explicit errors, exceptions, or corruption
- When uncertain, always choose Normal

You must respond in the exact format: "Label: [Anomaly/Normal]" followed by "Reason: [explanation]"."""

        response = self._call_llm(prompt, system_prompt)
        return self._parse_hdfs_anomaly_response(response, sequence_data)

    def analyze_root_cause(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'block_id' in data:
            return self.analyze_hdfs_anomaly(data)
        prompt = self._build_rca_prompt(data)
        response = self._call_llm(prompt)
        return self._parse_rca_response(response)

    def evaluate_anomaly_detection_performance(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, Any]:
        prompt = self._build_evaluation_prompt(predictions, ground_truth)
        system_prompt = """You are an expert in evaluating anomaly detection systems.
        Analyze the performance metrics, identify patterns in false positives/negatives,
        and provide recommendations for improvement."""
        response = self._call_llm(prompt, system_prompt)
        return self._parse_evaluation_response(response)

    def suggest_model_improvements(self, performance_data: Dict[str, Any]) -> List[str]:
        prompt = self._build_improvement_prompt(performance_data)
        system_prompt = """You are an ML expert specializing in log anomaly detection.
        Based on performance analysis, suggest specific improvements to enhance model accuracy."""
        response = self._call_llm(prompt, system_prompt)
        return self._parse_improvement_response(response)

    def analyze_anomaly(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self._build_anomaly_prompt(metrics)
        response = self._call_llm(prompt)
        return self._parse_anomaly_response(response)

    def suggest_remediation(self, issue: Dict[str, Any]) -> List[str]:
        prompt = self._build_remediation_prompt(issue)
        response = self._call_llm(prompt)
        return self._parse_remediation_response(response)

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.provider == LLMProvider.OPENAI:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(model=self.model, messages=messages)
            return response.choices[0].message.content

        elif self.provider == LLMProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.provider == LLMProvider.GROQ:
            if not self.client:
                return "Error: Groq client not initialized"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling Groq API: {str(e)}"

        elif self.provider == LLMProvider.HUGGINGFACE:
            import requests
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json={"inputs": prompt}
            )
            return response.json()[0]["generated_text"]

        elif self.provider == LLMProvider.OLLAMA:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            return response.json()["response"]

        return "Error: Unsupported provider"

    def _build_rca_prompt(self, data: Dict[str, Any]) -> str:
        return f"""Analyze the following system failure data and identify the root cause:

Anomalous Services: {data.get('anomalous_services', [])}
Metrics: {data.get('metrics', {})}
Dependencies: {data.get('dependencies', {})}
Logs: {data.get('logs', [])}
Components: {data.get('components', [])}

Provide:
1. Most likely root cause
2. Confidence score
3. Reasoning
4. Affected components"""

    def _build_rcaeval_rca_prompt(self, case_data: Dict[str, Any]) -> str:
        case_name = case_data.get('case_name', 'Unknown')
        fault_type = case_data.get('fault_type', 'Unknown')
        metrics_data = case_data.get('metrics', {})
        logs_data = case_data.get('logs', [])
        traces_data = case_data.get('traces', [])
        service_metadata = case_data.get('service_metadata', {})

        return f"""MICROSERVICE FAILURE ANALYSIS

Case Information:
- Case Name: {case_name}
- Fault Type: {fault_type}

Available Evidence:

1. METRICS DATA:
{self._format_metrics_for_prompt(metrics_data)}

2. LOG DATA:
{self._format_logs_for_prompt(logs_data)}

3. TRACE DATA:
{self._format_traces_for_prompt(traces_data)}

4. SERVICE METADATA:
{self._format_service_metadata_for_prompt(service_metadata)}

ANALYSIS REQUIREMENTS:

Based ONLY on the evidence provided above, perform a systematic root cause analysis:

1. EVIDENCE SUMMARY:
   - What concrete evidence do you observe?
   - What are the key symptoms vs. potential root causes?

2. FAULT PROPAGATION ANALYSIS:
   - How did the failure propagate through the system?
   - Which services were affected and in what order?

3. ROOT CAUSE IDENTIFICATION:
   - What is the most likely root cause based on evidence?
   - What alternative explanations are possible?
   - What evidence supports or contradicts each hypothesis?

4. CONFIDENCE ASSESSMENT:
   - How confident are you in this analysis (0-100%)?
   - What additional evidence would increase confidence?
   - What are the limitations of this analysis?

5. AFFECTED COMPONENTS:
   - Which services/components are directly affected?
   - Which are experiencing secondary effects?

IMPORTANT: Base your analysis ONLY on the provided evidence. Do not make assumptions based on the case name or fault type labels."""

    def _build_hdfs_anomaly_prompt(self, sequence_data: Dict[str, Any]) -> str:
        block_id = sequence_data.get('block_id', 'unknown')
        log_sequence = sequence_data.get('log_sequence', [])
        components = sequence_data.get('components', [])

        formatted_logs = [f"{i}. {entry}" for i, entry in enumerate(log_sequence, 1)]

        return f"""HDFS LOG SEQUENCE ANALYSIS

Block ID: {block_id}
Total Log Entries: {len(log_sequence)}
Components Involved: {', '.join(components)}

Log Sequence:
{chr(10).join(formatted_logs)}

ANALYSIS TASK:
Analyze this HDFS log sequence and determine if it represents normal or anomalous behavior.

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

    def _build_evaluation_prompt(self, predictions: List[Dict], ground_truth: List[Dict]) -> str:
        tp = fp = tn = fn = 0
        for pred, truth in zip(predictions, ground_truth):
            if pred.get('predicted_anomaly') and truth.get('actual_anomaly'):
                tp += 1
            elif pred.get('predicted_anomaly') and not truth.get('actual_anomaly'):
                fp += 1
            elif not pred.get('predicted_anomaly') and not truth.get('actual_anomaly'):
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f"""Evaluate this anomaly detection performance:

Metrics:
- True Positives: {tp}
- False Positives: {fp}
- True Negatives: {tn}
- False Negatives: {fn}
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1-Score: {f1:.3f}

Sample Misclassifications:
{self._format_misclassifications(predictions, ground_truth)}

Analyze:
1. Overall performance assessment
2. Patterns in false positives/negatives
3. Potential causes of misclassifications
4. Recommendations for improvement
5. Dataset quality issues (if any)"""

    def _build_improvement_prompt(self, performance_data: Dict[str, Any]) -> str:
        return f"""Based on this anomaly detection performance analysis:

{performance_data.get('evaluation_analysis', 'No analysis available')}

Performance Metrics:
- Precision: {performance_data.get('precision', 'N/A')}
- Recall: {performance_data.get('recall', 'N/A')}
- F1-Score: {performance_data.get('f1_score', 'N/A')}

Provide specific, actionable recommendations to improve model performance:
1. Data preprocessing improvements
2. Feature engineering suggestions
3. Model architecture changes
4. Training strategy modifications
5. Evaluation methodology enhancements

Focus on practical steps that can be implemented to address the identified issues."""

    def _build_anomaly_prompt(self, metrics: List[Dict[str, Any]]) -> str:
        return f"""Analyze the following system metrics for anomalies:

Metrics Data:
{json.dumps(metrics, indent=2)}

Identify:
1. Anomalous patterns
2. Potential root causes
3. Severity assessment
4. Recommended actions"""

    def _build_remediation_prompt(self, issue: Dict[str, Any]) -> str:
        return f"""Suggest remediation actions for this issue:

Issue Details:
{json.dumps(issue, indent=2)}

Provide:
1. Immediate actions
2. Short-term fixes
3. Long-term preventive measures
4. Monitoring recommendations"""

    def _format_metrics_for_prompt(self, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return "No metrics data available"
        formatted = []
        for service, service_metrics in metrics.items():
            formatted.append(f"Service: {service}")
            if isinstance(service_metrics, dict):
                for metric_name, values in service_metrics.items():
                    if isinstance(values, list) and values:
                        avg_val = sum(values) / len(values)
                        formatted.append(f"  {metric_name}: avg={avg_val:.2f}, max={max(values):.2f}, min={min(values):.2f}")
                    else:
                        formatted.append(f"  {metric_name}: {values}")
            formatted.append("")
        return "\n".join(formatted)

    def _format_logs_for_prompt(self, logs: List[str]) -> str:
        if not logs:
            return "No log data available"
        if len(logs) > 20:
            return f"Log entries (showing first 20 of {len(logs)}):\n" + "\n".join(logs[:20])
        return f"Log entries ({len(logs)} total):\n" + "\n".join(logs)

    def _format_traces_for_prompt(self, traces: List[Dict]) -> str:
        if not traces:
            return "No trace data available"
        formatted = []
        for i, trace in enumerate(traces[:10]):
            formatted.append(f"Trace {i+1}:")
            formatted.append(f"  Duration: {trace.get('duration', 'N/A')}")
            formatted.append(f"  Status: {trace.get('status', 'N/A')}")
            formatted.append(f"  Services: {trace.get('services', [])}")
            formatted.append("")
        if len(traces) > 10:
            formatted.append(f"... and {len(traces) - 10} more traces")
        return "\n".join(formatted)

    def _format_service_metadata_for_prompt(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return "No service metadata available"
        formatted = []
        for service, info in metadata.items():
            formatted.append(f"Service: {service}")
            if isinstance(info, dict):
                for key, value in info.items():
                    formatted.append(f"  {key}: {value}")
            else:
                formatted.append(f"  Info: {info}")
            formatted.append("")
        return "\n".join(formatted)

    def _format_misclassifications(self, predictions: List[Dict], ground_truth: List[Dict]) -> str:
        misclassifications = []
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred.get('predicted_anomaly') != truth.get('actual_anomaly'):
                pred_label = "Anomaly" if pred.get('predicted_anomaly') else "Normal"
                true_label = "Anomaly" if truth.get('actual_anomaly') else "Normal"
                block_id = pred.get('block_id', f'sample_{i}')
                confidence = pred.get('confidence', 0.0)
                misclassifications.append(
                    f"Block {block_id}: Predicted={pred_label}, Actual={true_label}, Confidence={confidence:.2f}"
                )
        if len(misclassifications) > 10:
            return "\n".join(misclassifications[:10]) + f"\n... and {len(misclassifications) - 10} more"
        return "\n".join(misclassifications)

    def _parse_rca_response(self, response: str) -> Dict[str, Any]:
        return {"analysis": response, "timestamp": None, "analysis_type": "generic_rca"}

    def _parse_rcaeval_rca_response(self, response: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        confidence = 0.0
        root_cause = "Unknown"
        for line in response.split('\n'):
            if 'confidence' in line.lower() and '%' in line:
                try:
                    confidence = float(line.split('%')[0].split()[-1]) / 100.0
                except Exception:
                    confidence = 0.7
            elif 'root cause' in line.lower():
                root_cause = line.split(':', 1)[-1].strip() if ':' in line else line.strip()
        return {
            "predicted_root_cause": root_cause,
            "confidence": confidence,
            "analysis": response,
            "case_name": case_data.get('case_name', 'Unknown'),
            "fault_type": case_data.get('fault_type', 'Unknown'),
            "timestamp": None,
            "analysis_type": "rcaeval_rca"
        }

    def _parse_hdfs_anomaly_response(self, response: str, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        predicted_anomaly = False
        confidence = 0.7
        response_lower = response.lower()

        if 'label: anomaly' in response_lower or 'classification: anomaly' in response_lower:
            predicted_anomaly = True
        elif 'label: normal' in response_lower or 'classification: normal' in response_lower:
            predicted_anomaly = False

        if 'confidence:' in response_lower:
            try:
                conf_line = [l for l in response.split('\n') if 'confidence:' in l.lower()][0]
                conf_str = conf_line.split(':')[1].strip().replace('%', '')
                confidence = float(conf_str) / 100.0 if conf_str.replace('.', '').isdigit() else 0.7
            except Exception:
                confidence = 0.7

        return {
            "predicted_anomaly": predicted_anomaly,
            "confidence": confidence,
            "analysis": response,
            "block_id": sequence_data.get('block_id', 'unknown'),
            "timestamp": None,
            "analysis_type": "hdfs_anomaly"
        }

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        return {"evaluation_analysis": response, "analysis_type": "performance_evaluation", "timestamp": None}

    def _parse_improvement_response(self, response: str) -> List[str]:
        improvements = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                improvements.append(line.lstrip('-*123456789. '))
        return improvements if improvements else [response]

    def _parse_anomaly_response(self, response: str) -> Dict[str, Any]:
        return {"analysis": response, "timestamp": None, "analysis_type": "anomaly_analysis"}

    def _parse_remediation_response(self, response: str) -> List[str]:
        actions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                actions.append(line.lstrip('-*123456789. '))
        return actions if actions else [response]
