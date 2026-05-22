from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["model"] == "Extra Trees Classifier"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "tenure_months": 21,
        "monthly_charge": 62.45,
        "total_charges": 1365.15,
        "support_tickets": 1,
        "late_payments": 1,
        "avg_monthly_usage_gb": 142.87,
        "contract_type": "mensual",
        "payment_method": "credito",
        "internet_service": "movil",
        "has_streaming": 1,
        "has_security_pack": 1,
        "num_products": 3,
        "region": "norte",
        "customer_age": 52,
        "is_promo": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability_churn" in data
    assert "risk_level" in data
    assert "message" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability_churn"] <= 1
    assert data["risk_level"] in ["bajo", "medio", "alto"]