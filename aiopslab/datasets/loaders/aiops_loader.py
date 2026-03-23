from pathlib import Path
import pandas as pd
from typing import Dict, Any


class AiopsLoader:
    def __init__(self, base_path: Path, **kwargs):
        self.base_path = base_path / "aiops_challenge"
        self.config = kwargs

    def load(self) -> Dict[str, Any]:
        return {
            "metrics": self._load_metrics(),
            "traces": self._load_traces(),
            "logs": self._load_logs(),
            "labels": self._load_labels()
        }

    def _load_metrics(self):
        metrics_file = self.base_path / "metrics.csv"
        if metrics_file.exists():
            return pd.read_csv(metrics_file)
        return pd.DataFrame()

    def _load_traces(self):
        traces_file = self.base_path / "traces.json"
        if traces_file.exists():
            import json
            with open(traces_file) as f:
                return json.load(f)
        return []

    def _load_logs(self):
        logs_file = self.base_path / "application.log"
        if logs_file.exists():
            with open(logs_file) as f:
                return f.readlines()
        return []

    def _load_labels(self):
        labels_file = self.base_path / "labels.csv"
        if labels_file.exists():
            return pd.read_csv(labels_file)
        return pd.DataFrame()
