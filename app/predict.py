from __future__ import annotations
from app.schemas import PredictionRequest, PredictionResponse
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
        "Brick": payload.bricks,
        "Neighborhood": payload.neighborhood
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)

def predict_price(model, payload: PredictionRequest) -> float:
    data = build_model_input(payload)
    model = load_model()
    predicted_price = float(model.predict(data)[0])

    return predicted_price