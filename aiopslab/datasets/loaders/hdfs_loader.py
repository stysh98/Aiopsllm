from pathlib import Path
import pandas as pd
import re
from typing import Dict, Any, List


class HdfsLoader:
    def __init__(self, base_path: Path, **kwargs):
        if str(base_path).startswith('/aiopslab'):
            self.base_path = Path("datasets") / "hdfs"
        else:
            self.base_path = base_path / "hdfs"
        self.config = kwargs
        self.log_format = "<Date> <Time> <Pid> <Level> <Component> <Content>"

    def load(self) -> Dict[str, Any]:
        labels = self._load_labels()

        log_file = self.base_path / "HDFS.log"
        total_lines = 0
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f)

        sequences = self._create_sample_sequences(labels, max_sequences=1000)

        return {
            "raw_logs": [],
            "parsed_logs": [],
            "labels": labels,
            "sequences": sequences,
            "anomaly_stats": self._get_anomaly_stats(labels),
            "dataset_info": {
                "name": "HDFS_v1",
                "source": "LogHub",
                "format": self.log_format,
                "total_logs": total_lines,
                "total_sequences": len(labels) if not labels.empty else 0,
                "anomalous_sequences": len(labels[labels['Label'] == 1]) if not labels.empty else 0,
                "sampled_sequences": len(sequences)
            }
        }

    def _create_sample_sequences(self, labels: pd.DataFrame, max_sequences: int = 1000) -> List[Dict[str, Any]]:
        if labels.empty:
            return []

        normal_labels = labels[labels['Label'] == 0]
        anomalous_labels = labels[labels['Label'] == 1]

        max_anomalous = min(max_sequences // 2, len(anomalous_labels))
        max_normal = min(max_sequences - max_anomalous, len(normal_labels))

        sample_labels = pd.concat([
            anomalous_labels.head(max_anomalous),
            normal_labels.head(max_normal)
        ]).sample(frac=1).reset_index(drop=True)

        print(f"Sampling {len(sample_labels)} sequences ({max_anomalous} anomalous, {max_normal} normal)")

        log_file = self.base_path / "HDFS.log"
        if not log_file.exists():
            return []

        target_blocks = set(sample_labels['BlockId'].values)
        block_logs = {}

        print("Parsing logs for sampled sequences...")
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000000 == 0:
                    print(f"Processed {line_num:,} lines...")
                line = line.strip()
                if not line:
                    continue
                block_id = self._extract_block_id(line)
                if block_id and block_id in target_blocks:
                    if block_id not in block_logs:
                        block_logs[block_id] = []
                    parsed = self._parse_single_log(line, line_num)
                    if parsed:
                        block_logs[block_id].append(parsed)

        sequences = []
        for _, row in sample_labels.iterrows():
            block_id = row['BlockId']
            label = int(row['Label'])
            logs = block_logs.get(block_id, [])
            if not logs:
                continue
            sequences.append({
                'block_id': block_id,
                'logs': logs,
                'log_count': len(logs),
                'label': label,
                'is_anomaly': bool(label),
                'templates': [log.get('content', '') for log in logs],
                'components': list(set(log.get('component', '') for log in logs if log.get('component')))
            })

        print(f"Created {len(sequences)} sequences with log data")
        return sequences

    def _parse_single_log(self, log_line: str, line_id: int) -> Dict[str, Any]:
        pattern = r'(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([^:]+):\s+(.*)'
        match = re.match(pattern, log_line)
        if match:
            date, time, pid, level, component, content = match.groups()
            return {
                'line_id': line_id,
                'date': date,
                'time': time,
                'pid': pid,
                'level': level,
                'component': component.strip(),
                'content': content.strip(),
                'block_id': self._extract_block_id(content),
                'raw_line': log_line
            }
        return {
            'line_id': line_id,
            'raw_line': log_line,
            'content': log_line,
            'block_id': self._extract_block_id(log_line),
            'malformed': True
        }

    def _load_logs(self) -> List[str]:
        log_file = self.base_path / "HDFS.log"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        for alt_name in ["HDFS_2k.log", "HDFS_5k.log", "hdfs.log"]:
            alt_file = self.base_path / alt_name
            if alt_file.exists():
                with open(alt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.readlines()
        print(f"Warning: No HDFS log file found in {self.base_path}")
        return []

    def _load_labels(self) -> pd.DataFrame:
        labels_file = self.base_path / "anomaly_label.csv"
        if labels_file.exists():
            try:
                labels_df = pd.read_csv(labels_file)
                if 'BlockId' in labels_df.columns and 'Label' in labels_df.columns:
                    labels_df['Label'] = labels_df['Label'].map({
                        'Normal': 0, 'Anomaly': 1, 0: 0, 1: 1
                    }).fillna(0).astype(int)
                    return labels_df
                elif len(labels_df.columns) >= 2:
                    labels_df.columns = ['BlockId', 'Label']
                    if labels_df['Label'].dtype == 'object':
                        labels_df['Label'] = labels_df['Label'].map({
                            'Normal': 0, 'Anomaly': 1, 'normal': 0, 'anomaly': 1, 0: 0, 1: 1
                        }).fillna(0).astype(int)
                    return labels_df
            except Exception as e:
                print(f"Error loading labels: {e}")
        print(f"Warning: No anomaly labels found in {self.base_path}")
        return pd.DataFrame(columns=['BlockId', 'Label'])

    def _parse_logs(self, logs: List[str]) -> List[Dict[str, Any]]:
        parsed_logs = []
        pattern = r'(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([^:]+):\s+(.*)'
        for i, log_line in enumerate(logs):
            log_line = log_line.strip()
            if not log_line:
                continue
            match = re.match(pattern, log_line)
            if match:
                date, time, pid, level, component, content = match.groups()
                parsed_logs.append({
                    'line_id': i,
                    'date': date,
                    'time': time,
                    'pid': pid,
                    'level': level,
                    'component': component.strip(),
                    'content': content.strip(),
                    'block_id': self._extract_block_id(content),
                    'raw_line': log_line
                })
            else:
                parsed_logs.append({
                    'line_id': i,
                    'raw_line': log_line,
                    'content': log_line,
                    'block_id': self._extract_block_id(log_line),
                    'malformed': True
                })
        return parsed_logs

    def _extract_block_id(self, content: str) -> str:
        match = re.search(r'(blk_-?\d+)', content)
        return match.group(1) if match else None

    def _create_sequences(self, parsed_logs: List[Dict], labels: pd.DataFrame) -> List[Dict[str, Any]]:
        if labels.empty:
            return []
        block_groups = {}
        for log in parsed_logs:
            block_id = log.get('block_id')
            if block_id:
                if block_id not in block_groups:
                    block_groups[block_id] = []
                block_groups[block_id].append(log)
        labels_dict = dict(zip(labels['BlockId'], labels['Label']))
        sequences = []
        for block_id, logs in block_groups.items():
            label = labels_dict.get(block_id, 0)
            sequences.append({
                'block_id': block_id,
                'logs': sorted(logs, key=lambda x: x['line_id']),
                'log_count': len(logs),
                'label': int(label),
                'is_anomaly': bool(int(label)),
                'templates': [log.get('content', '') for log in logs],
                'components': list(set(log.get('component', '') for log in logs if log.get('component')))
            })
        return sequences

    def _get_anomaly_stats(self, labels: pd.DataFrame) -> Dict[str, Any]:
        if labels.empty:
            return {"total": 0, "normal": 0, "anomalous": 0, "anomaly_rate": 0.0}
        total = len(labels)
        anomalous = len(labels[labels['Label'] == 1])
        return {
            "total": total,
            "normal": total - anomalous,
            "anomalous": anomalous,
            "anomaly_rate": anomalous / total if total > 0 else 0.0
        }
