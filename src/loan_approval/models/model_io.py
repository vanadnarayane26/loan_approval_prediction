import joblib
from pathlib import Path

def save_model(model, path:str):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model, path)
    
def load_model(path:str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)