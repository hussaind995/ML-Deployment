from fastapi.testclient import TestClient 

from app.main import app

client = TestClient(app)

def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_input() -> None:
    payload = {
                "SqFt": 10000,
                "Bedrooms": 50,
                "Bathrooms": 40,
                "Offers": 5,
                "Brick": "Yes",
                "Neighborhood": "ABCD"
            }
    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert body["predicted_price"] >= 0

def test_predict_endpoint_with_invalid_input() -> None:
    payload = {
                "SqFt": 10000,
                "Bedrooms": 50,
                "Bathrooms": 40,
                "Offers": 5,
                "Brick": 5,
                "Neighborhood": "North"
            }

    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 422
