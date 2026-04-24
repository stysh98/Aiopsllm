import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from aiopslab.agents.llm_agent import LLMAgent
from aiopslab.datasets.adapter import DatasetAdapter


class AIOpsLab:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        llm_config = self.config.get("llm", {})
        llm_config.update({
            "provider": os.getenv("LLM_PROVIDER", "groq"),
            "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        })
        self.llm_agent = LLMAgent(llm_config)
        self.dataset_adapter = DatasetAdapter(self.config.get("datasets", {}))
        self.experiments = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def setup_cluster(self, cluster_name: str = "aiopslab"):
        print("Kubernetes cluster setup not available in this version")
        return None

    def load_dataset(self, dataset_name: str, **kwargs):
        return self.dataset_adapter.load(dataset_name, **kwargs)

    def run_experiment(self, experiment_config: Dict[str, Any]):
        from aiopslab.core.experiment import Experiment
        exp = Experiment(self, experiment_config)
        return exp.run()

    def cleanup(self):
        print("Cleanup completed")
