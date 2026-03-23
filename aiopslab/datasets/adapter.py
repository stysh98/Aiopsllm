from pathlib import Path
from typing import Dict, Any, Optional
import importlib


class DatasetAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets_dir = Path(config.get("base_path", "datasets"))
        self.loaders = {}
        self._register_loaders()
    
    def _register_loaders(self):
        self.loaders = {
            "hdfs": "aiopslab.datasets.loaders.hdfs_loader.HdfsLoader",
            "rcaeval": "aiopslab.datasets.loaders.rcaeval_loader.RcaevalLoader",
            "aiops_challenge": "aiopslab.datasets.loaders.aiops_loader.AiopsLoader",
        }
    
    def load(self, dataset_name: str, **kwargs) -> Any:
        loader_module = self.loaders.get(dataset_name)
        
        if not loader_module:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        module_path, class_name = loader_module.rsplit(".", 1)
        module = importlib.import_module(module_path)
        loader_class = getattr(module, class_name)
        
        loader = loader_class(self.datasets_dir, **kwargs)
        return loader.load()
    
    def list_available(self) -> list:
        return list(self.loaders.keys())
