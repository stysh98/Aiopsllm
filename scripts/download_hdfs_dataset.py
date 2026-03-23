#!/usr/bin/env python3
import os
import requests
import zipfile
from pathlib import Path
import pandas as pd


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


def download_hdfs_dataset():
    datasets_dir = Path("datasets/hdfs")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "LogHub GitHub (2k sample)",
            "log_url": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
            "label_url": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/anomaly_label.csv"
        },
        {
            "name": "Alternative GitHub source",
            "log_url": "https://github.com/logpai/loghub/raw/master/HDFS/HDFS_2k.log",
            "label_url": "https://github.com/logpai/loghub/raw/master/HDFS/anomaly_label.csv"
        },
        {
            "name": "Zenodo",
            "log_url": "https://zenodo.org/records/3227177/files/HDFS.log",
            "label_url": "https://zenodo.org/records/3227177/files/anomaly_label.csv"
        }
    ]

    log_file = datasets_dir / "HDFS.log"
    labels_file = datasets_dir / "anomaly_label.csv"
    success = False

    for source in sources:
        print(f"\n=== Trying {source['name']} ===")
        try:
            if not log_file.exists():
                download_file(source["log_url"], log_file)
            if not labels_file.exists():
                download_file(source["label_url"], labels_file)
            if log_file.exists() and labels_file.exists():
                print(f"Successfully downloaded from {source['name']}")
                success = True
                break
        except Exception as e:
            print(f"Failed to download from {source['name']}: {e}")
            if log_file.exists():
                log_file.unlink()
            if labels_file.exists():
                labels_file.unlink()
            continue

    if not success:
        print("\nAll download sources failed. Creating sample dataset instead...")
        create_sample_dataset()
        return

    verify_dataset(datasets_dir)


def verify_dataset(datasets_dir: Path):
    print("\nVerifying dataset...")

    log_file = datasets_dir / "HDFS.log"
    labels_file = datasets_dir / "anomaly_label.csv"

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return False
    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return False

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_lines = f.readlines()
        print(f"Log file: {len(log_lines):,} lines")
        print("Sample log entries:")
        for i, line in enumerate(log_lines[:3]):
            print(f"  {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading log file: {e}")
        return False

    try:
        labels_df = pd.read_csv(labels_file)
        total = len(labels_df)
        label_col = next((c for c in ['Label', 'label', 'anomaly', 'Anomaly'] if c in labels_df.columns), None)
        if label_col:
            anomalous = len(labels_df[labels_df[label_col] == 1])
            print(f"Labels file: {total:,} sequences, {anomalous:,} anomalous ({anomalous/total:.1%})")
        else:
            print(f"Labels file: {total:,} sequences")
            print(f"Columns: {list(labels_df.columns)}")
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return False

    print("Dataset verification complete!")
    return True


def create_sample_dataset():
    datasets_dir = Path("datasets/hdfs")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    sample_logs = [
        "081109 203615 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203615 143 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating",
        "081109 203615 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862907 src: /10.250.19.102:54107 dest: /10.250.19.102:50010",
        "081109 203615 144 ERROR dfs.DataNode$DataXceiver: DataXceiver error processing unknown operation src: /10.250.19.102:54108 dest: /10.250.19.102:50010",
        "081109 203615 144 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862907 terminating",
        "081109 203616 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862908 src: /10.250.19.103:54109 dest: /10.250.19.103:50010",
        "081109 203616 145 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862908 terminating",
        "081109 203617 146 ERROR dfs.DataNode$DataXceiver: IOException in BlockReceiver constructor. Cause is java.io.IOException: Premature EOF from inputStream",
        "081109 203617 146 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862909 src: /10.250.19.104:54110 dest: /10.250.19.104:50010",
        "081109 203617 146 ERROR dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862909 Interrupted."
    ]

    sample_labels = pd.DataFrame({
        'BlockId': [
            'blk_-1608999687919862906',
            'blk_-1608999687919862907',
            'blk_-1608999687919862908',
            'blk_-1608999687919862909'
        ],
        'Label': [0, 1, 0, 1]
    })

    with open(datasets_dir / "HDFS.log", 'w') as f:
        f.write('\n'.join(sample_logs))

    sample_labels.to_csv(datasets_dir / "anomaly_label.csv", index=False)

    print("Sample dataset created for testing")
    print(f"   - {len(sample_logs)} log entries")
    print(f"   - {len(sample_labels)} block sequences")
    print(f"   - {len(sample_labels[sample_labels['Label'] == 1])} anomalous sequences")


def download_full_hdfs_dataset():
    print("Downloading full HDFS_v1 dataset from Zenodo...")

    datasets_dir = Path("datasets/hdfs")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    hdfs_url = "https://zenodo.org/api/records/3227177/files/HDFS_1.tar.gz/content"
    hdfs_file = datasets_dir / "HDFS_1.tar.gz"

    try:
        if not hdfs_file.exists():
            print("Downloading HDFS_v1 dataset (161MB)...")
            download_file(hdfs_url, hdfs_file)
        else:
            print("HDFS_1.tar.gz already exists, skipping download...")

        print("Extracting HDFS dataset...")
        import tarfile
        with tarfile.open(hdfs_file, 'r:gz') as tar:
            tar.extractall(datasets_dir)

        extracted_files = list(datasets_dir.glob("*"))
        print(f"Extracted files: {[f.name for f in extracted_files if f.is_file()]}")

        hdfs_log = None
        labels_file = None
        for file_path in datasets_dir.rglob("*"):
            if file_path.is_file():
                if "HDFS" in file_path.name and file_path.suffix == ".log":
                    hdfs_log = file_path
                elif "anomaly_label" in file_path.name or "label" in file_path.name:
                    labels_file = file_path

        if hdfs_log and hdfs_log.name != "HDFS.log":
            hdfs_log.rename(datasets_dir / "HDFS.log")
        if labels_file and labels_file.name != "anomaly_label.csv":
            labels_file.rename(datasets_dir / "anomaly_label.csv")

        hdfs_file.unlink()
        verify_dataset(datasets_dir)
        return True

    except Exception as e:
        print(f"Error downloading full HDFS dataset: {e}")
        print("Falling back to sample dataset...")
        create_sample_dataset()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HDFS dataset")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset for testing")
    parser.add_argument("--full", action="store_true", help="Attempt to download full dataset")
    args = parser.parse_args()

    if args.sample:
        create_sample_dataset()
    elif args.full:
        download_full_hdfs_dataset()
    else:
        download_hdfs_dataset()
