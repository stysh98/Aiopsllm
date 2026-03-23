import re
from typing import Dict, Any
from datetime import datetime


class Experiment:
    def __init__(self, framework, config: Dict[str, Any]):
        self.framework = framework
        self.config = config
        self.name = config.get("name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results = {}

    def run(self):
        print(f"Running experiment: {self.name}")
        if "datasets" in self.config:
            return self._run_multi_dataset_experiment()
        return self._run_single_dataset_experiment()

    def _run_single_dataset_experiment(self):
        dataset = self.framework.load_dataset(
            self.config["dataset"]["name"],
            **self.config["dataset"].get("params", {})
        )
        if self.config.get("deploy_workload"):
            self._deploy_workload()
        if self.config.get("anomaly_detection"):
            self._run_anomaly_detection(dataset)
        if self.config.get("rca"):
            self._run_rca(dataset)
        return self.results

    def _run_multi_dataset_experiment(self):
        print("Running multi-dataset AIOps experiment...")
        datasets = {}

        for dataset_key, dataset_config in self.config["datasets"].items():
            print(f"Loading {dataset_key} dataset...")
            datasets[dataset_key] = self.framework.load_dataset(
                dataset_config["name"],
                **dataset_config.get("params", {})
            )

        if self.config.get("anomaly_detection", {}).get("enabled"):
            ad_dataset_key = self.config["anomaly_detection"].get("dataset", "hdfs")
            if ad_dataset_key in datasets:
                print(f"\n=== Phase 1: Anomaly Detection using {ad_dataset_key.upper()} ===")
                self._run_anomaly_detection(datasets[ad_dataset_key])

        if self.config.get("rca", {}).get("enabled"):
            rca_dataset_key = self.config["rca"].get("dataset", "rcaeval")
            if rca_dataset_key in datasets:
                print(f"\n=== Phase 2: Root Cause Analysis using {rca_dataset_key.upper()} ===")
                self._run_rca(datasets[rca_dataset_key])

        if self.config.get("cross_analysis", {}).get("enabled"):
            print("\n=== Phase 3: Cross-Dataset Analysis ===")
            self._run_cross_analysis(datasets)

        self.results["experiment_type"] = "multi_dataset"
        self.results["datasets_used"] = list(datasets.keys())
        self.results["dataset_info"] = {
            key: dataset.get("dataset_info", {})
            for key, dataset in datasets.items()
        }
        return self.results

    def _run_cross_analysis(self, datasets):
        print("Performing cross-dataset analysis...")
        hdfs_results = self.results.get("anomaly_detection", {})
        rcaeval_results = self.results.get("rca", {})

        if not hdfs_results or not rcaeval_results:
            print("Warning: Missing results from previous phases for cross-analysis")
            return

        cross_analysis = self._analyze_pattern_similarities(hdfs_results, rcaeval_results)
        unified_insights = self._generate_unified_insights(hdfs_results, rcaeval_results)

        self.results["cross_analysis"] = {
            "pattern_similarities": cross_analysis,
            "unified_insights": unified_insights,
            "comparison_summary": {
                "hdfs_anomalies_analyzed": len(hdfs_results.get("predictions", [])),
                "rcaeval_failures_analyzed": len(rcaeval_results.get("rca_results", [])),
                "common_patterns_found": len(cross_analysis.get("common_patterns", [])),
                "integration_recommendations": unified_insights.get("recommendations", [])
            }
        }

    def _analyze_pattern_similarities(self, hdfs_results, rcaeval_results):
        hdfs_patterns = self._extract_hdfs_patterns(hdfs_results)
        rcaeval_patterns = self._extract_rcaeval_patterns(rcaeval_results)

        similarity_prompt = f"""Analyze the patterns between these two AIOps datasets:

HDFS Anomaly Patterns:
{hdfs_patterns}

RCAEval Failure Patterns:
{rcaeval_patterns}

Identify:
1. Common failure patterns across both datasets
2. Complementary insights (what each dataset reveals)
3. Integration opportunities for better AIOps
4. Patterns that could improve cross-system analysis

Provide structured analysis with specific examples."""

        try:
            similarity_analysis = self.framework.llm_agent._call_llm(
                similarity_prompt,
                "You are an expert in AIOps pattern analysis across different system types."
            )
            return {
                "analysis": similarity_analysis,
                "hdfs_patterns": hdfs_patterns,
                "rcaeval_patterns": rcaeval_patterns,
                "common_patterns": self._extract_common_patterns(similarity_analysis)
            }
        except Exception as e:
            print(f"Error in pattern similarity analysis: {e}")
            return {"error": str(e)}

    def _generate_unified_insights(self, hdfs_results, rcaeval_results):
        unified_prompt = f"""Based on analysis of both HDFS anomaly detection and RCAEval root cause analysis:

HDFS Results Summary:
- Anomalies detected: {len(hdfs_results.get('predictions', []))}
- Detection accuracy: F1-Score {hdfs_results.get('evaluation', {}).get('f1_score', 'N/A')}
- Key anomaly types: Block corruption, replication issues, network problems

RCAEval Results Summary:
- Failure cases analyzed: {len(rcaeval_results.get('rca_results', []))}
- Systems covered: Microservices (Online Boutique, Sock Shop, Train Ticket)
- Fault types: CPU, Memory, Disk, Network delays, Code-level faults

Generate unified insights for:
1. Complete AIOps pipeline recommendations
2. How HDFS anomaly detection complements microservice RCA
3. Integration strategies for production deployment
4. Monitoring and alerting improvements
5. Future research directions

Provide actionable recommendations for enterprise AIOps systems."""

        try:
            unified_analysis = self.framework.llm_agent._call_llm(
                unified_prompt,
                "You are an expert AIOps architect designing enterprise-scale systems."
            )
            return {
                "analysis": unified_analysis,
                "recommendations": self._extract_recommendations(unified_analysis),
                "integration_strategies": self._extract_integration_strategies(unified_analysis)
            }
        except Exception as e:
            print(f"Error in unified insights generation: {e}")
            return {"error": str(e)}

    def _extract_hdfs_patterns(self, hdfs_results):
        patterns = []
        for pred in hdfs_results.get("predictions", [])[:5]:
            patterns.append(f"Block {pred.get('block_id', 'unknown')}: {pred.get('analysis', '')[:200]}...")
        return "\n".join(patterns)

    def _extract_rcaeval_patterns(self, rcaeval_results):
        patterns = []
        for result in rcaeval_results.get("rca_results", [])[:5]:
            case_id = result.get('case_id', 'unknown')
            system = result.get('system', 'unknown')
            fault_type = result.get('fault_type', 'unknown')
            analysis = result.get('rca_analysis', {}).get('analysis', '')
            patterns.append(f"Case {case_id} ({system}, {fault_type}): {analysis[:200]}...")
        return "\n".join(patterns)

    def _extract_common_patterns(self, analysis_text):
        lines = analysis_text.split('\n')
        patterns = [l.strip() for l in lines if 'pattern' in l.lower() or 'common' in l.lower()]
        return patterns[:5]

    def _extract_recommendations(self, analysis_text):
        lines = analysis_text.split('\n')
        recs = [l.strip() for l in lines if any(w in l.lower() for w in ['recommend', 'suggest', 'should', 'improve'])]
        return recs[:10]

    def _extract_integration_strategies(self, analysis_text):
        lines = analysis_text.split('\n')
        strategies = [l.strip() for l in lines if any(w in l.lower() for w in ['integration', 'combine', 'unified', 'together'])]
        return strategies[:5]

    def _deploy_workload(self):
        print("Deploying workload to kind cluster...")

    def _run_anomaly_detection(self, dataset):
        print("Running HDFS anomaly detection with LLM analysis...")

        dataset_info = dataset.get('dataset_info', {})
        sequences = dataset.get('sequences', [])

        print(f"Dataset: {dataset_info.get('name', 'Unknown')}")
        print(f"Total sequences: {dataset_info.get('total_sequences', 0):,}")
        print(f"Anomalous sequences: {dataset_info.get('anomalous_sequences', 0):,}")
        print(f"Anomaly rate: {dataset.get('anomaly_stats', {}).get('anomaly_rate', 0):.2%}")

        if not sequences:
            print("Warning: No sequences found for analysis")
            return

        config = self.config.get('anomaly_detection', {}).get('params', {})
        sample_size = config.get('sample_size', 50)
        max_normal_samples = config.get('max_normal_samples', 25)
        focus_on_anomalous = config.get('focus_on_anomalous', True)

        normal_sequences = [s for s in sequences if not s.get('is_anomaly', False)]
        anomalous_sequences = [s for s in sequences if s.get('is_anomaly', True)]

        print(f"Available: {len(normal_sequences):,} normal, {len(anomalous_sequences):,} anomalous")

        sample_sequences = []
        if focus_on_anomalous and anomalous_sequences:
            anomalous_sample_size = min(sample_size // 2, len(anomalous_sequences))
            normal_sample_size = min(max_normal_samples, sample_size - anomalous_sample_size, len(normal_sequences))
            sample_sequences.extend(anomalous_sequences[:anomalous_sample_size])
            sample_sequences.extend(normal_sequences[:normal_sample_size])
        else:
            sample_sequences = sequences[:sample_size]

        sample_normal = sum(1 for s in sample_sequences if not s.get('is_anomaly', False))
        sample_anomalous = len(sample_sequences) - sample_normal
        print(f"Analyzing {len(sample_sequences)} sequences:")
        print(f"  - {sample_normal} normal sequences")
        print(f"  - {sample_anomalous} anomalous sequences")

        predictions = []
        ground_truth = []

        for i, sequence in enumerate(sample_sequences):
            label = 'Anomalous' if sequence.get('is_anomaly') else 'Normal'
            print(f"Analyzing sequence {i+1}/{len(sample_sequences)}: Block {sequence.get('block_id')} ({label})")
            try:
                llm_result = self.framework.llm_agent.analyze_hdfs_anomaly(sequence)
                predictions.append({
                    'block_id': sequence.get('block_id'),
                    'predicted_anomaly': self._extract_prediction_from_analysis(llm_result),
                    'confidence': self._extract_confidence_from_analysis(llm_result),
                    'analysis': llm_result.get('analysis', '')
                })
                ground_truth.append({
                    'block_id': sequence.get('block_id'),
                    'actual_anomaly': sequence.get('is_anomaly', False)
                })
            except Exception as e:
                print(f"Error analyzing sequence {i+1}: {e}")
                continue

        if not predictions:
            print("No successful predictions made")
            return

        print("Evaluating LLM performance...")
        evaluation = self.framework.llm_agent.evaluate_anomaly_detection_performance(predictions, ground_truth)

        print("Generating improvement suggestions...")
        improvements = self.framework.llm_agent.suggest_model_improvements(evaluation)

        self.results["anomaly_detection"] = {
            "dataset_info": dataset_info,
            "sampling_strategy": {
                "total_available": len(sequences),
                "sample_size": len(sample_sequences),
                "normal_sampled": sample_normal,
                "anomalous_sampled": sample_anomalous,
                "focus_on_anomalous": focus_on_anomalous
            },
            "predictions": predictions,
            "ground_truth": ground_truth,
            "evaluation": evaluation,
            "improvements": improvements
        }

    def _extract_prediction_from_analysis(self, llm_result: dict) -> bool:
        analysis = llm_result.get('analysis', '').lower()

        classification_match = re.search(r'classification:\s*(anomaly|normal)', analysis)
        if classification_match:
            return classification_match.group(1) == 'anomaly'

        decision_patterns = [
            r'this\s+(?:is|represents)\s+(?:an\s+)?anomaly',
            r'classify\s+(?:this\s+)?as\s+anomaly',
            r'conclusion:\s*anomaly',
            r'decision:\s*anomaly',
            r'result:\s*anomaly'
        ]
        for pattern in decision_patterns:
            if re.search(pattern, analysis):
                return True

        normal_patterns = [
            r'this\s+(?:is|represents)\s+normal',
            r'classify\s+(?:this\s+)?as\s+normal',
            r'conclusion:\s*normal',
            r'decision:\s*normal',
            r'result:\s*normal',
            r'standard\s+(?:hdfs\s+)?operations?',
            r'routine\s+(?:hdfs\s+)?operations?'
        ]
        for pattern in normal_patterns:
            if re.search(pattern, analysis):
                return False

        anomaly_keywords = [
            'error', 'failure', 'failed', 'corruption', 'corrupted', 'timeout',
            'exception', 'crash', 'disconnect', 'unavailable', 'unreachable',
            'checksum', 'mismatch', 'invalid', 'missing', 'lost'
        ]
        normal_keywords = [
            'successful', 'completed', 'allocated', 'stored', 'received',
            'normal', 'standard', 'routine', 'expected', 'typical'
        ]

        anomaly_score = sum(len(re.findall(kw, analysis)) for kw in anomaly_keywords)
        normal_score = sum(len(re.findall(kw, analysis)) for kw in normal_keywords)

        return anomaly_score > normal_score + 2

    def _extract_confidence_from_analysis(self, llm_result: dict) -> float:
        analysis = llm_result.get('analysis', '')

        confidence_match = re.search(r'confidence:\s*(\d+)%', analysis.lower())
        if confidence_match:
            return float(confidence_match.group(1)) / 100.0

        confidence_patterns = [
            r'confidence[:\s]*(\d+)%',
            r'(\d+)%\s*confidence',
            r'confidence[:\s]*(\d+)/100',
            r'confidence[:\s]*0\.(\d+)',
            r'confidence\s+level:\s*(\d+)%'
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                value = float(match.group(1))
                if value <= 1.0:
                    return value
                elif value <= 100:
                    return value / 100.0

        analysis_lower = analysis.lower()
        high_conf_indicators = ['clear evidence', 'obvious', 'definitely', 'certainly', 'strong indication', 'conclusive', 'unambiguous']
        low_conf_indicators = ['uncertain', 'unclear', 'possibly', 'might', 'could be', 'potential', 'suspected', 'ambiguous', 'inconclusive']

        high_conf_score = sum(1 for i in high_conf_indicators if i in analysis_lower)
        low_conf_score = sum(1 for i in low_conf_indicators if i in analysis_lower)

        if high_conf_score > low_conf_score:
            return 0.8
        elif low_conf_score > high_conf_score:
            return 0.4
        return 0.7 if len(analysis) > 200 else 0.5

    def _run_rca(self, dataset):
        print("Running RCA with LLM agent...")
        if dataset.get('dataset_info', {}).get('name') == 'RCAEval':
            self._run_rcaeval_rca(dataset)
        else:
            self._run_hdfs_rca(dataset)

    def _run_rcaeval_rca(self, dataset):
        print("Running RCAEval root cause analysis...")

        dataset_info = dataset.get('dataset_info', {})
        sequences = dataset.get('sequences', [])

        print(f"Dataset: {dataset_info.get('name', 'Unknown')}")
        print(f"Total failure cases: {dataset_info.get('total_cases', 0):,}")
        print(f"Systems: {', '.join(dataset_info.get('systems', []))}")
        print(f"Fault types: {', '.join(dataset_info.get('fault_types', []))}")

        if not sequences:
            print("Warning: No failure cases found for analysis")
            return

        config = self.config.get('rca', {}).get('params', {})
        sample_size = config.get('sample_size', 20)
        prioritize_multi_source = config.get('prioritize_multi_source', True)

        if prioritize_multi_source:
            multi_source_cases = [s for s in sequences if s.get('has_logs') or s.get('has_traces')]
            metrics_only_cases = [s for s in sequences if not s.get('has_logs') and not s.get('has_traces')]
            multi_source_sample = min(sample_size * 2 // 3, len(multi_source_cases))
            metrics_only_sample = min(sample_size - multi_source_sample, len(metrics_only_cases))
            sample_sequences = multi_source_cases[:multi_source_sample] + metrics_only_cases[:metrics_only_sample]
        else:
            sample_sequences = sequences[:sample_size]

        multi_source_count = sum(1 for s in sample_sequences if s.get('has_logs') or s.get('has_traces'))
        print(f"Analyzing {len(sample_sequences)} failure cases:")
        print(f"  - {multi_source_count} multi-source cases (with logs/traces)")
        print(f"  - {len(sample_sequences) - multi_source_count} metrics-only cases")

        rca_results = []
        for i, case in enumerate(sample_sequences):
            case_id = case.get('case_id', f'case_{i}')
            system = case.get('system', 'unknown')
            fault_type = case.get('fault_type', 'unknown')
            print(f"RCA {i+1}/{len(sample_sequences)}: {case_id} ({system}, {fault_type})")
            try:
                rca_analysis = self.framework.llm_agent.analyze_rcaeval_failure(case)
                remediation = self.framework.llm_agent.suggest_remediation(rca_analysis)
                rca_results.append({
                    'case_id': case_id,
                    'system': system,
                    'fault_type': fault_type,
                    'actual_root_cause': case.get('root_cause_service'),
                    'rca_analysis': rca_analysis,
                    'remediation_suggestions': remediation
                })
            except Exception as e:
                print(f"Error analyzing case {i+1}: {e}")
                continue

        self.results["rca"] = {
            "dataset_type": "RCAEval",
            "total_cases": len(sequences),
            "analyzed_cases": len(sample_sequences),
            "successful_analyses": len(rca_results),
            "sampling_strategy": {
                "prioritize_multi_source": prioritize_multi_source,
                "multi_source_analyzed": multi_source_count,
                "metrics_only_analyzed": len(sample_sequences) - multi_source_count
            },
            "rca_results": rca_results
        }

    def _run_hdfs_rca(self, dataset):
        sequences = dataset.get('sequences', [])
        anomalous_sequences = [s for s in sequences if s.get('is_anomaly', True)]

        if not anomalous_sequences:
            print("No anomalous sequences found for RCA")
            return

        print(f"Analyzing {len(anomalous_sequences)} anomalous sequences for root cause...")
        sample_size = min(5, len(anomalous_sequences))
        rca_results = []

        for i, sequence in enumerate(anomalous_sequences[:sample_size]):
            print(f"RCA for anomalous sequence {i+1}/{sample_size}: Block {sequence.get('block_id')}")
            rca_data = {
                'block_id': sequence.get('block_id'),
                'logs': sequence.get('logs', []),
                'components': sequence.get('components', []),
                'log_count': sequence.get('log_count', 0),
                'templates': sequence.get('templates', [])
            }
            rca_analysis = self.framework.llm_agent.analyze_root_cause(rca_data)
            remediation = self.framework.llm_agent.suggest_remediation(rca_analysis)
            rca_results.append({
                'block_id': sequence.get('block_id'),
                'rca_analysis': rca_analysis,
                'remediation_suggestions': remediation
            })

        self.results["rca"] = {
            "dataset_type": "HDFS",
            "total_anomalous_sequences": len(anomalous_sequences),
            "analyzed_sequences": sample_size,
            "rca_results": rca_results
        }
