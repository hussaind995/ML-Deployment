from __future__ import annotations
from functools import lru_cache
from pathlib import Path 
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "model.joblib"

@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)