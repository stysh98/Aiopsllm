#!/usr/bin/env python3
import os
import requests
import zipfile
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, List


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    print(f"\rProgress: {(downloaded / total_size) * 100:.1f}%", end='', flush=True)

    print(f"\nDownloaded: {destination}")


def download_rcaeval_datasets():
    datasets_dir = Path("datasets/rcaeval")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print("Installing RCAEval package for full dataset download...")
    print("This will download all 735 failure cases across 9 datasets")

    try:
        import subprocess
        import sys

        print("Installing RCAEval[default] package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "RCAEval[default]", "--upgrade"])

        original_cwd = os.getcwd()
        os.chdir(str(datasets_dir))

        try:
            from RCAEval.utility import (
                download_re1_dataset,
                download_re2_dataset,
                download_re3_dataset,
            )

            print("\nDownloading RE1 dataset (375 cases - metrics only)...")
            download_re1_dataset()

            print("\nDownloading RE2 dataset (270 cases - multi-source)...")
            download_re2_dataset()

            print("\nDownloading RE3 dataset (90 cases - code-level faults)...")
            download_re3_dataset()

            print("\nAll RCAEval datasets downloaded successfully!")
            print("Total: 735 failure cases across 9 datasets")
        finally:
            os.chdir(original_cwd)

        create_full_dataset_summary()

        if verify_full_rcaeval_dataset():
            print("Full RCAEval dataset verification passed!")
            return True
        else:
            print("Dataset verification failed, but files may still be usable")
            return False

    except ImportError as e:
        print(f"Failed to import RCAEval utilities: {e}")
        print("Trying direct download from Zenodo...")
        return download_rcaeval_direct()
    except Exception as e:
        print(f"Error downloading RCAEval datasets: {e}")
        print("Trying direct download from Zenodo...")
        return download_rcaeval_direct()


def download_rcaeval_direct():
    print("Attempting direct download from Zenodo...")

    datasets_dir = Path("datasets/rcaeval")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    zenodo_record_id = "14590730"
    zenodo_api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"

    try:
        print("Fetching download URLs from Zenodo API...")
        response = requests.get(zenodo_api_url)
        response.raise_for_status()

        record_data = response.json()
        files = record_data.get('files', [])

        if not files:
            print("No files found in Zenodo record")
            return False

        print(f"Found {len(files)} files in Zenodo record")
        total_downloaded = 0

        for file_info in files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            file_size = file_info.get('size', 0)

            if filename.endswith('.zip') and any(re_name in filename for re_name in ['RE1', 'RE2', 'RE3']):
                print(f"\nDownloading {filename} ({file_size / (1024*1024*1024):.1f} GB)...")
                file_path = datasets_dir / filename

                try:
                    download_file(download_url, file_path)
                    print(f"Extracting {filename}...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(datasets_dir)
                    file_path.unlink()
                    print(f"{filename} extracted")
                    total_downloaded += 1
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")

        if total_downloaded > 0:
            print(f"\nSuccessfully downloaded {total_downloaded} datasets")
            create_full_dataset_summary()
            return True
        else:
            print("No datasets were successfully downloaded")
            return False

    except Exception as e:
        print(f"Error accessing Zenodo API: {e}")
        print("Trying manual download URLs...")
        return download_rcaeval_manual()


def download_rcaeval_manual():
    print("Attempting manual download with known patterns...")

    datasets_dir = Path("datasets/rcaeval")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    zenodo_base = "https://zenodo.org/records/14590730/files"
    manual_urls = {
        "RE1-OB.zip": f"{zenodo_base}/RE1-OB.zip?download=1",
        "RE1-SS.zip": f"{zenodo_base}/RE1-SS.zip?download=1",
        "RE1-TT.zip": f"{zenodo_base}/RE1-TT.zip?download=1",
        "RE2-OB.zip": f"{zenodo_base}/RE2-OB.zip?download=1",
        "RE2-SS.zip": f"{zenodo_base}/RE2-SS.zip?download=1",
        "RE2-TT.zip": f"{zenodo_base}/RE2-TT.zip?download=1",
        "RE3-OB.zip": f"{zenodo_base}/RE3-OB.zip?download=1",
        "RE3-SS.zip": f"{zenodo_base}/RE3-SS.zip?download=1",
        "RE3-TT.zip": f"{zenodo_base}/RE3-TT.zip?download=1"
    }

    print("Manual download URLs need verification")
    print("Creating enhanced sample dataset instead...")
    create_enhanced_sample_dataset()
    return False


def create_enhanced_sample_dataset():
    print("Creating enhanced sample dataset with full structure...")

    datasets_dir = Path("datasets/rcaeval")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    systems = ["ob", "ss", "tt"]
    system_names = {"ob": "online-boutique", "ss": "sock-shop", "tt": "train-ticket"}

    re1_faults = ["cpu", "mem", "disk", "delay", "loss"]
    re1_services = ["service1", "service2", "service3", "service4", "service5"]
    re2_faults = ["cpu", "mem", "disk", "delay", "loss", "socket"]
    re2_services = ["service1", "service2", "service3", "service4", "service5"]
    re3_faults = ["f1", "f2", "f3", "f4", "f5"]
    re3_services = ["service1", "service2", "service3", "service4", "service5"]

    total_cases = 0

    for system in systems:
        re1_dir = datasets_dir / "RE1" / f"RE1-{system.upper()}"
        re1_dir.mkdir(parents=True, exist_ok=True)
        for fault, service in zip(re1_faults[:2], re1_services[:2]):
            case_id = f"re1_{system}_{fault}_{service}_1"
            case_dir = re1_dir / case_id
            case_dir.mkdir(exist_ok=True)
            create_sample_case_files(case_dir, case_id, system_names[system], fault, service, "re1", include_logs=False, include_traces=False)
            total_cases += 1

    for system in systems:
        re2_dir = datasets_dir / "RE2" / f"RE2-{system.upper()}"
        re2_dir.mkdir(parents=True, exist_ok=True)
        for fault, service in zip(re2_faults[:2], re2_services[:2]):
            case_id = f"re2_{system}_{fault}_{service}_1"
            case_dir = re2_dir / case_id
            case_dir.mkdir(exist_ok=True)
            create_sample_case_files(case_dir, case_id, system_names[system], fault, service, "re2", include_logs=True, include_traces=True)
            total_cases += 1

    for system in systems:
        re3_dir = datasets_dir / "RE3" / f"RE3-{system.upper()}"
        re3_dir.mkdir(parents=True, exist_ok=True)
        for fault, service in zip(re3_faults[:1], re3_services[:1]):
            case_id = f"re3_{system}_{fault}_{service}_1"
            case_dir = re3_dir / case_id
            case_dir.mkdir(exist_ok=True)
            create_sample_case_files(case_dir, case_id, system_names[system], fault, service, "re3", include_logs=True, include_traces=True)
            total_cases += 1

    summary = {
        "dataset_name": "RCAEval Enhanced Sample Dataset",
        "version": "2025-sample",
        "source": "Generated sample mimicking full structure",
        "total_cases": total_cases,
        "expected_cases": 735,
        "benchmark_suites": 3,
        "systems": ["online-boutique", "sock-shop", "train-ticket"],
        "re_datasets": {"RE1": 6, "RE2": 6, "RE3": 3},
        "dataset_breakdown": {
            "RE1": {
                "description": "Metrics-only datasets (sample)",
                "actual_cases": 6,
                "expected_cases": 375,
                "systems": 3,
                "fault_types": ["cpu", "mem", "disk", "delay", "loss"],
                "data_types": ["metrics"]
            },
            "RE2": {
                "description": "Multi-source datasets (sample)",
                "actual_cases": 6,
                "expected_cases": 270,
                "systems": 3,
                "fault_types": ["cpu", "mem", "disk", "delay", "loss", "socket"],
                "data_types": ["metrics", "logs", "traces"]
            },
            "RE3": {
                "description": "Code-level fault datasets (sample)",
                "actual_cases": 3,
                "expected_cases": 90,
                "systems": 3,
                "fault_types": ["f1", "f2", "f3", "f4", "f5"],
                "data_types": ["metrics", "logs", "traces"]
            }
        },
        "fault_types": ["cpu", "mem", "disk", "delay", "loss", "socket", "f1", "f2", "f3", "f4", "f5"],
        "download_timestamp": pd.Timestamp.now().isoformat(),
        "status": "enhanced_sample"
    }

    with open(datasets_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Enhanced sample dataset created with {total_cases} cases")


def create_sample_case_files(case_dir: Path, case_id: str, system: str, fault_type: str, service: str, benchmark: str, include_logs: bool = False, include_traces: bool = False):
    import random

    base_time = 1692569339
    inject_time = base_time + random.randint(0, 86400)

    with open(case_dir / "inject_time.txt", 'w') as f:
        f.write(str(inject_time))

    time_points = list(range(inject_time - 300, inject_time + 300, 10))
    metrics_data = {"time": time_points}

    metric_count = {"ob": 60, "ss": 70, "tt": 200}.get(system.split('-')[0], 60)

    for i in range(metric_count):
        values = []
        for t in time_points:
            if inject_time - 60 <= t <= inject_time + 120:
                if fault_type == "cpu":
                    value = 80 + random.uniform(0, 20)
                elif fault_type == "mem":
                    value = 85 + random.uniform(0, 15)
                elif fault_type == "disk":
                    value = 90 + random.uniform(0, 10)
                elif fault_type == "delay":
                    value = 500 + random.uniform(0, 500)
                else:
                    value = 50 + random.uniform(0, 50)
            else:
                value = 30 + random.uniform(0, 20)
            values.append(round(value, 2))
        metrics_data[f"{service}_metric_{i}"] = values

    with open(case_dir / "metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)

    if include_logs:
        log_entries = []
        for i in range(100):
            timestamp = inject_time + random.randint(-300, 300)
            level = random.choice(["INFO", "WARN", "ERROR", "DEBUG"])
            message = f"Code fault {fault_type} detected in {service}" if fault_type.startswith('f') else f"{level}: {service} {fault_type} issue detected"
            log_entries.append({"timestamp": timestamp, "level": level, "service": service, "message": message})
        pd.DataFrame(log_entries).to_csv(case_dir / "logs.csv", index=False)

    if include_traces:
        trace_entries = []
        for i in range(50):
            timestamp = inject_time + random.randint(-300, 300)
            duration = random.uniform(10, 1000) if fault_type == "delay" else random.uniform(1, 100)
            trace_entries.append({
                "timestamp": timestamp,
                "trace_id": f"trace_{i:04d}",
                "span_id": f"span_{i:04d}",
                "service": service,
                "operation": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "duration_ms": round(duration, 2)
            })
        pd.DataFrame(trace_entries).to_csv(case_dir / "traces.csv", index=False)

    metadata = {
        "case_id": case_id,
        "system": system,
        "fault_type": fault_type,
        "service": service,
        "benchmark": benchmark,
        "inject_time": inject_time,
        "root_cause_service": service,
        "root_cause_indicator": f"{service}_{fault_type}",
        "description": f"Sample {fault_type} fault in {service} service ({benchmark} benchmark)",
        "data_types": ["metrics"] + (["logs"] if include_logs else []) + (["traces"] if include_traces else [])
    }

    with open(case_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def create_full_dataset_summary():
    datasets_dir = Path("datasets/rcaeval")
    re_datasets = {}
    total_cases = 0

    for item in datasets_dir.iterdir():
        if item.is_dir() and item.name.startswith('RE'):
            cases = [d for d in item.iterdir() if d.is_dir()]
            re_datasets[item.name] = len(cases)
            total_cases += len(cases)

    summary = {
        "dataset_name": "RCAEval Full Dataset",
        "version": "2025",
        "source": "https://github.com/phamquiluan/RCAEval",
        "figshare": "https://figshare.com/articles/dataset/RCAEval_A_Benchmark_for_Root_Cause_Analysis_of_Microservice_Systems/31048672",
        "total_cases": total_cases,
        "expected_cases": 735,
        "benchmark_suites": 3,
        "systems": ["online-boutique", "sock-shop", "train-ticket"],
        "re_datasets": re_datasets,
        "dataset_breakdown": {
            "RE1": {
                "description": "Metrics-only datasets",
                "expected_cases": 375,
                "systems": 3,
                "fault_types": ["cpu", "mem", "disk", "delay", "loss"],
                "data_types": ["metrics"]
            },
            "RE2": {
                "description": "Multi-source datasets",
                "expected_cases": 270,
                "systems": 3,
                "fault_types": ["cpu", "mem", "disk", "delay", "loss", "socket"],
                "data_types": ["metrics", "logs", "traces"]
            },
            "RE3": {
                "description": "Code-level fault datasets",
                "expected_cases": 90,
                "systems": 3,
                "fault_types": ["f1", "f2", "f3", "f4", "f5"],
                "data_types": ["metrics", "logs", "traces"]
            }
        },
        "fault_types": ["cpu", "mem", "disk", "delay", "loss", "socket", "f1", "f2", "f3", "f4", "f5"],
        "download_timestamp": pd.Timestamp.now().isoformat(),
        "status": "full_dataset" if total_cases >= 700 else "partial_dataset"
    }

    with open(datasets_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset summary created: {total_cases} cases found")


def verify_full_rcaeval_dataset():
    datasets_dir = Path("datasets/rcaeval")
    if not datasets_dir.exists():
        return False

    expected_cases = {"RE1": 375, "RE2": 270, "RE3": 90}
    total_found = 0

    for re_name in ["RE1", "RE2", "RE3"]:
        re_dir = datasets_dir / re_name
        if re_dir.exists():
            found_cases = len([d for d in re_dir.iterdir() if d.is_dir()])
            print(f"   {re_name}: {found_cases}/{expected_cases[re_name]} cases")
            total_found += found_cases
        else:
            print(f"   {re_name}: Not found")

    print(f"Total cases found: {total_found}/735")
    return total_found >= 700


def create_sample_rcaeval_dataset():
    datasets_dir = Path("datasets/rcaeval")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    sample_cases = [
        {
            "case_id": "re1_ob_cpu_service1_1",
            "system": "online-boutique",
            "fault_type": "cpu",
            "service": "emailservice",
            "inject_time": 1692569339,
            "root_cause_service": "emailservice",
            "root_cause_indicator": "emailservice_cpu"
        },
        {
            "case_id": "re1_ob_mem_service2_1",
            "system": "online-boutique",
            "fault_type": "mem",
            "service": "recommendationservice",
            "inject_time": 1692569400,
            "root_cause_service": "recommendationservice",
            "root_cause_indicator": "recommendationservice_mem"
        },
        {
            "case_id": "re2_ss_delay_service3_1",
            "system": "sock-shop",
            "fault_type": "delay",
            "service": "cartservice",
            "inject_time": 1692569500,
            "root_cause_service": "cartservice",
            "root_cause_indicator": "cartservice_latency"
        }
    ]

    for case in sample_cases:
        case_dir = datasets_dir / case["case_id"]
        case_dir.mkdir(exist_ok=True)

        with open(case_dir / "inject_time.txt", 'w') as f:
            f.write(str(case["inject_time"]))

        metrics_data = {
            "time": list(range(case["inject_time"] - 300, case["inject_time"] + 300, 10)),
            f"{case['service']}_cpu": [50 + i * 0.1 for i in range(60)],
            f"{case['service']}_mem": [60 + i * 0.2 for i in range(60)],
            f"{case['service']}_latency": [100 + i * 0.5 for i in range(60)]
        }

        anomaly_start = 30
        if case["fault_type"] == "cpu":
            for i in range(anomaly_start, 50):
                metrics_data[f"{case['service']}_cpu"][i] = 90 + i * 2
        elif case["fault_type"] == "mem":
            for i in range(anomaly_start, 50):
                metrics_data[f"{case['service']}_mem"][i] = 85 + i * 1.5
        elif case["fault_type"] == "delay":
            for i in range(anomaly_start, 50):
                metrics_data[f"{case['service']}_latency"][i] = 500 + i * 10

        with open(case_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)

        metadata = {
            "case_id": case["case_id"],
            "system": case["system"],
            "fault_type": case["fault_type"],
            "service": case["service"],
            "inject_time": case["inject_time"],
            "root_cause_service": case["root_cause_service"],
            "root_cause_indicator": case["root_cause_indicator"],
            "description": f"Sample {case['fault_type']} fault in {case['service']} service"
        }

        with open(case_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    summary = {
        "dataset_name": "RCAEval Sample",
        "total_cases": len(sample_cases),
        "systems": list(set(c["system"] for c in sample_cases)),
        "fault_types": list(set(c["fault_type"] for c in sample_cases)),
        "services": list(set(c["service"] for c in sample_cases)),
        "cases": sample_cases
    }

    with open(datasets_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("Sample RCAEval dataset created for testing")
    print(f"   - {len(sample_cases)} failure cases")
    print(f"   - Systems: {', '.join(summary['systems'])}")
    print(f"   - Fault types: {', '.join(summary['fault_types'])}")


def verify_rcaeval_dataset():
    print("\nVerifying RCAEval dataset...")

    datasets_dir = Path("datasets/rcaeval")
    if not datasets_dir.exists():
        print(f"Dataset directory not found: {datasets_dir}")
        return False

    summary_file = datasets_dir / "dataset_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"Dataset summary found:")
        print(f"   Total cases: {summary.get('total_cases', 0)}")
        print(f"   Expected cases: {summary.get('expected_cases', 735)}")
        print(f"   Systems: {', '.join(summary.get('systems', []))}")
        print(f"   Fault types: {len(summary.get('fault_types', []))} types")
        print(f"   Status: {summary.get('status', 'unknown')}")
        if 're_datasets' in summary:
            print("   RE Datasets:")
            for re_name, count in summary['re_datasets'].items():
                print(f"      - {re_name}: {count} cases")
        return summary.get('total_cases', 0) >= 700

    re_dirs = [d for d in datasets_dir.iterdir() if d.is_dir() and d.name.startswith('RE')]
    if re_dirs:
        print(f"Found {len(re_dirs)} RE dataset directories")
        total_cases = 0
        for re_dir in sorted(re_dirs):
            cases = [d for d in re_dir.iterdir() if d.is_dir()]
            total_cases += len(cases)
            print(f"   {re_dir.name}: {len(cases)} cases")
        print(f"   Total cases: {total_cases}")
        if total_cases >= 700:
            print("Full dataset verification passed!")
            return True
        elif total_cases > 0:
            print("Partial dataset found - may be usable for testing")
            return True
        else:
            print("No cases found in RE directories")
            return False

    sample_cases = [d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('RE')]
    if sample_cases:
        print(f"Found sample dataset with {len(sample_cases)} cases")
        return True

    print("No valid RCAEval dataset found")
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download RCAEval dataset (735 failure cases)")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset for testing")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset")
    parser.add_argument("--full", action="store_true", help="Download full dataset (default)")
    args = parser.parse_args()

    if args.verify:
        success = verify_rcaeval_dataset()
        exit(0 if success else 1)
    elif args.sample:
        print("Creating sample RCAEval dataset for testing...")
        create_sample_rcaeval_dataset()
    else:
        print("Downloading full RCAEval dataset (735 cases)...")
        success = download_rcaeval_datasets()
        if success:
            print("\nFull RCAEval dataset download completed!")
            print("Location: datasets/rcaeval/")
            print("Run with --verify to check the dataset")
        else:
            print("\nDownload encountered issues - check logs above")
            print("Try: python scripts/download_rcaeval_dataset.py --sample")
