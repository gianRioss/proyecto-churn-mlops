from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app


client = TestClient(app)


VALID_PAYLOAD = {
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
    "is_promo": 1,
}


def get_valid_payload():
    """Devuelve una copia independiente del payload válido."""
    return VALID_PAYLOAD.copy()


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["model"] == "Extra Trees Classifier"
    assert data["model_version"] == "1.0.0"
    assert data["api_version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["inference_test"] is True
    assert data["test_prediction"] in [0, 1]
    assert 0 <= data["test_probability"] <= 1
    assert data["model_name"] == "Extra Trees Classifier"


def test_predict_endpoint():
    response = client.post("/predict", json=get_valid_payload())

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability_churn" in data
    assert "risk_level" in data
    assert "message" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability_churn"] <= 1
    assert data["risk_level"] in ["bajo", "medio", "alto"]


def test_predict_missing_required_field():
    payload = get_valid_payload()
    payload.pop("contract_type")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_incorrect_type():
    payload = get_valid_payload()
    payload["tenure_months"] = "texto_invalido"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_numeric_value_out_of_range():
    payload = get_valid_payload()
    payload["monthly_charge"] = -10

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_category():
    payload = get_valid_payload()
    payload["contract_type"] = "banana"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    data = response.json()
    assert data["detail"][0]["type"] == "literal_error"


def test_predict_invalid_binary_value():
    payload = get_valid_payload()
    payload["has_streaming"] = 2

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_model_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)

    response = client.post("/predict", json=get_valid_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "El modelo de predicción no está disponible"
    )


def test_predict_inference_failure(monkeypatch):
    class BrokenModel:
        def predict(self, input_df):
            raise RuntimeError("Error simulado de inferencia")

    monkeypatch.setattr(api_module, "model", BrokenModel())

    response = client.post("/predict", json=get_valid_payload())

    assert response.status_code == 500
    assert response.json()["detail"] == "No se pudo generar la predicción"


def test_health_model_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "El modelo no está disponible"


def test_health_inference_failure(monkeypatch):
    class BrokenModel:
        def predict(self, input_df):
            raise RuntimeError("Error simulado en healthcheck")

    monkeypatch.setattr(api_module, "model", BrokenModel())

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "La API está activa, pero el modelo no supera la prueba de inferencia"
    )