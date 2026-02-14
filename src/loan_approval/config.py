import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        self.config_dict = self.config_load()
        
        self.data = self.config_dict.get("data", {})
        self.model = self.config_dict.get("model", {})  
        self.artifacts = self.config_dict.get("artifacts", {})
        self.inference = self.config_dict.get("inference", {})
        
    def config_load(self):
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return config_dict