from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any, List


class RcaevalLoader:
    def __init__(self, base_path: Path, **kwargs):
        if str(base_path).startswith('/aiopslab'):
            self.base_path = Path("datasets") / "rcaeval"
        else:
            self.base_path = base_path / "rcaeval"
        self.config = kwargs
        self.supported_systems = ["online-boutique", "sock-shop", "train-ticket"]
        self.supported_fault_types = ["cpu", "mem", "disk", "delay", "loss", "socket", "f1", "f2", "f3", "f4", "f5"]

    def load(self) -> Dict[str, Any]:
        cases = self._load_cases()
        sequences = self._create_sequences(cases)
        stats = self._get_dataset_stats(cases)
        return {
            "raw_cases": cases,
            "sequences": sequences,
            "dataset_stats": stats,
            "dataset_info": {
                "name": "RCAEval",
                "source": "Figshare/GitHub",
                "description": "Benchmark for Root Cause Analysis of Microservice Systems",
                "total_cases": len(cases),
                "total_sequences": len(sequences),
                "systems": list(set(case.get('system', 'unknown') for case in cases)),
                "fault_types": list(set(case.get('fault_type', 'unknown') for case in cases)),
                "services": list(set(case.get('service', 'unknown') for case in cases))
            }
        }

    def _load_cases(self) -> List[Dict[str, Any]]:
        cases = []
        summary_file = self.base_path / "dataset_summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"Loading from dataset summary: {summary.get('total_cases', 0)} cases")

            if 're_datasets' in summary and summary['re_datasets']:
                for re_name in summary['re_datasets']:
                    re_dir = self.base_path / re_name
                    if re_dir.exists():
                        re_cases = self._load_re_dataset(re_dir)
                        cases.extend(re_cases)
                        print(f"   {re_name}: {len(re_cases)} cases loaded")
            elif 'cases' in summary:
                for case_info in summary['cases']:
                    case_data = self._load_single_case(case_info['case_id'])
                    if case_data:
                        cases.append(case_data)
        else:
            print("Scanning for RE dataset directories...")
            re_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith('RE')]
            if re_dirs:
                print(f"   Found {len(re_dirs)} RE datasets")
                for re_dir in sorted(re_dirs):
                    re_cases = self._load_re_dataset(re_dir)
                    cases.extend(re_cases)
                    print(f"   {re_dir.name}: {len(re_cases)} cases loaded")
            else:
                print("Scanning for individual case directories...")
                for case_dir in self.base_path.iterdir():
                    if case_dir.is_dir():
                        case_data = self._load_single_case(case_dir.name, case_dir)
                        if case_data:
                            cases.append(case_data)

        print(f"Loaded {len(cases)} failure cases from RCAEval dataset")
        if len(cases) >= 700:
            print("Full dataset loaded successfully!")
        elif len(cases) > 0:
            print("Partial dataset loaded - may be sample or incomplete download")
        else:
            print("No cases loaded - check dataset directory")

        return cases

    def _load_re_dataset(self, re_dir: Path) -> List[Dict[str, Any]]:
        cases = []
        for service_fault_dir in re_dir.iterdir():
            if service_fault_dir.is_dir():
                for instance_dir in service_fault_dir.iterdir():
                    if instance_dir.is_dir():
                        case_id = f"{re_dir.name}_{service_fault_dir.name}_{instance_dir.name}"
                        case_data = self._load_single_case(case_id, instance_dir)
                        if case_data:
                            cases.append(case_data)
        return cases

    def _load_single_case(self, case_id: str, case_dir: Path = None) -> Dict[str, Any]:
        if case_dir is None:
            case_dir = self.base_path / case_id
        if not case_dir.exists():
            return None

        case_data = {"case_id": case_id, "case_dir": str(case_dir)}
        case_data.update(self._parse_case_id(case_id))

        inject_time_file = case_dir / "inject_time.txt"
        if inject_time_file.exists():
            with open(inject_time_file, 'r') as f:
                case_data["inject_time"] = int(f.read().strip())

        data_file = case_dir / "data.csv"
        if data_file.exists():
            metrics_df = pd.read_csv(data_file)
            metrics_data = {col: metrics_df[col].tolist() for col in metrics_df.columns}
            case_data["metrics"] = metrics_data
            case_data["metric_names"] = [k for k in metrics_data if k != 'time']

        logs_file = case_dir / "logs.csv"
        if logs_file.exists():
            case_data["logs"] = pd.read_csv(logs_file)

        traces_file = case_dir / "traces.csv"
        if traces_file.exists():
            case_data["traces"] = pd.read_csv(traces_file)

        metadata_file = case_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                case_data.update(json.load(f))

        case_data["root_cause_service"] = case_data.get("service")
        case_data["root_cause_indicator"] = f"{case_data.get('service', 'unknown')}_{case_data.get('fault_type', 'unknown')}"
        return case_data

    def _parse_case_id(self, case_id: str) -> Dict[str, Any]:
        parts = case_id.split('_')
        info = {}

        if len(parts) >= 3:
            benchmark_system = parts[0]
            if '-' in benchmark_system:
                benchmark_part, system_abbrev = benchmark_system.split('-', 1)
                info["benchmark"] = benchmark_part.lower()
                system_mapping = {"OB": "online-boutique", "SS": "sock-shop", "TT": "train-ticket"}
                info["system"] = system_mapping.get(system_abbrev, system_abbrev.lower())
            else:
                info["benchmark"] = "unknown"
                info["system"] = "unknown"

            if len(parts) >= 2:
                service_fault = parts[1]
                fault_type = parts[2] if len(parts) > 2 else "unknown"
                known_fault_types = ["cpu", "mem", "disk", "delay", "loss", "socket", "f1", "f2", "f3", "f4", "f5"]

                if fault_type in known_fault_types:
                    info["service"] = service_fault
                    info["fault_type"] = fault_type
                else:
                    for ft in known_fault_types:
                        if service_fault.endswith(f"_{ft}"):
                            info["service"] = service_fault[:-len(f"_{ft}")]
                            info["fault_type"] = ft
                            break
                    else:
                        info["service"] = service_fault
                        info["fault_type"] = "unknown"

            if len(parts) > 2:
                try:
                    info["instance"] = int(parts[-1])
                except (ValueError, IndexError):
                    info["instance"] = 1
        else:
            info = {"benchmark": "unknown", "system": "unknown", "fault_type": "unknown", "service": "unknown"}

        return info

    def _create_sequences(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sequences = []
        for case in cases:
            sequence = {
                "case_id": case["case_id"],
                "system": case.get("system", "unknown"),
                "fault_type": case.get("fault_type", "unknown"),
                "service": case.get("service", "unknown"),
                "inject_time": case.get("inject_time", 0),
                "root_cause_service": case.get("root_cause_service", "unknown"),
                "root_cause_indicator": case.get("root_cause_indicator", "unknown"),
                "is_failure": True,
                "has_metrics": "metrics" in case,
                "has_logs": "logs" in case,
                "has_traces": "traces" in case,
                "benchmark": case.get("benchmark", "unknown")
            }

            if "metrics" in case:
                sequence["metrics_data"] = case["metrics"]
                sequence["metric_names"] = case.get("metric_names", [])
                sequence["time_series_length"] = len(case["metrics"].get("time", []))

            if "logs" in case:
                sequence["logs_data"] = case["logs"]
                sequence["log_count"] = len(case["logs"])

            if "traces" in case:
                sequence["traces_data"] = case["traces"]
                sequence["trace_count"] = len(case["traces"])

            sequence["summary"] = self._create_case_summary(case)
            sequences.append(sequence)

        return sequences

    def _create_case_summary(self, case: Dict[str, Any]) -> str:
        parts = [
            f"Failure Case: {case['case_id']}",
            f"System: {case.get('system', 'unknown')}",
            f"Fault Type: {case.get('fault_type', 'unknown')}",
            f"Affected Service: {case.get('service', 'unknown')}",
            f"Root Cause Service: {case.get('root_cause_service', 'unknown')}",
            f"Root Cause Indicator: {case.get('root_cause_indicator', 'unknown')}",
        ]
        if "inject_time" in case:
            parts.append(f"Fault Injection Time: {case['inject_time']}")
        if "metrics" in case:
            parts.append(f"Available Metrics: {len(case.get('metric_names', []))}")
        if "logs" in case:
            parts.append(f"Log Entries: {len(case['logs'])}")
        if "traces" in case:
            parts.append(f"Trace Records: {len(case['traces'])}")
        return "\n".join(parts)

    def _get_dataset_stats(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cases:
            return {}

        systems = [c.get("system", "unknown") for c in cases]
        fault_types = [c.get("fault_type", "unknown") for c in cases]
        services = [c.get("service", "unknown") for c in cases]
        benchmarks = [c.get("benchmark", "unknown") for c in cases]

        metrics_count = sum(1 for c in cases if "metrics" in c)
        logs_count = sum(1 for c in cases if "logs" in c)
        traces_count = sum(1 for c in cases if "traces" in c)
        n = len(cases)

        stats = {
            "total_cases": n,
            "expected_full_dataset": 735,
            "dataset_completeness": n / 735 if n <= 735 else 1.0,
            "systems": {
                "unique": list(set(systems)),
                "distribution": {s: systems.count(s) for s in set(systems)},
                "count": len(set(systems))
            },
            "fault_types": {
                "unique": list(set(fault_types)),
                "distribution": {ft: fault_types.count(ft) for ft in set(fault_types)},
                "count": len(set(fault_types))
            },
            "services": {"unique": list(set(services)), "count": len(set(services))},
            "benchmarks": {
                "unique": list(set(benchmarks)),
                "distribution": {b: benchmarks.count(b) for b in set(benchmarks)},
                "count": len(set(benchmarks))
            },
            "data_availability": {
                "metrics": metrics_count,
                "logs": logs_count,
                "traces": traces_count,
                "metrics_percentage": (metrics_count / n) * 100,
                "logs_percentage": (logs_count / n) * 100,
                "traces_percentage": (traces_count / n) * 100
            },
            "data_types": {
                "metrics_only": sum(1 for c in cases if "metrics" in c and "logs" not in c and "traces" not in c),
                "multi_source": sum(1 for c in cases if "metrics" in c and ("logs" in c or "traces" in c)),
                "with_logs": logs_count,
                "with_traces": traces_count,
                "complete_telemetry": sum(1 for c in cases if "metrics" in c and "logs" in c and "traces" in c)
            },
            "benchmark_breakdown": {
                "RE1": sum(1 for c in cases if c.get("benchmark") == "re1"),
                "RE2": sum(1 for c in cases if c.get("benchmark") == "re2"),
                "RE3": sum(1 for c in cases if c.get("benchmark") == "re3")
            },
            "quality_indicators": {
                "has_inject_times": sum(1 for c in cases if "inject_time" in c),
                "has_root_cause_info": sum(1 for c in cases if "root_cause_service" in c),
                "complete_metadata": sum(1 for c in cases if all(k in c for k in ["system", "fault_type", "service", "inject_time"]))
            }
        }
        return stats
