from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self):
        if not self.data_path.exists(): 
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df