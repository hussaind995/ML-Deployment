from __future__ import annotations
from fastapi import FastAPI
from app.model_loader import load_model 
from app.predict import predict_price
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="House Price Prediction", version="1.0.0")

@app.get("/health")
def health_check() -> dict[str, str]:
    load_model()
    return {"status": "ok"}

@app.get("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = load_model()
    predicted_price = predict_price(model=model, payload=payload)

    return PredictionResponse(predicted_price=predicted_price)
