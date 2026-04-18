from __future__ import annotations
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import load_model
import pandas as pd 

MODEL_FEATURES = [
    "SqFt",
    "Bedrooms",
    "Bathrooms",
    "Offers",
    "Brick",
    "Neighborhood"
]


def build_model_input(payload: PredictionRequest) -> pd.DataFrame:
    row = {
        "SqFt": payload.sqft,
        "Bedrooms": payload.bedrooms,
        "Bathrooms": payload.bathrooms,
        "Offers": payload.offers,
        "Brick": payload.brick,
        "Neighborhood": payload.neighborhood
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)

def predict_price(model, payload: PredictionRequest) -> float:
    data = build_model_input(payload)
    predicted_price = float(model.predict(data)[0])

    return predicted_price