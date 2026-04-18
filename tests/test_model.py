from __future__ import annotations
from app.schemas import PredictionRequest
from app.predict import build_model_input, predict_price
from app.model_loader import load_model

def sample_payload() -> PredictionRequest:
    return PredictionRequest(
        SqFt = 10000,
        Bedrooms = 50,
        Bathrooms = 40,
        Offers = 5,
        Brick = "Yes",
        Neighborhood = "ABCD"
    )

def test_model_load() -> None:
    model = load_model()
    assert model is not None

def test_build_model_input_gives_expected_output() -> None:
    payload = sample_payload()
    data = build_model_input(payload)

    assert data.shape == (1, 6)
    assert data.columns.to_list() == [
                                    "SqFt",
                                    "Bedrooms",
                                    "Bathrooms",
                                    "Offers",
                                    "Brick",
                                    "Neighborhood"
                                ]

def test_predict_prices_gives_expected_output() -> None: 
    payload = sample_payload()
    model = load_model()

    predicted_price = predict_price(model, payload)

    assert predicted_price >= 0

