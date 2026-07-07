import logging
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_churn_extra_trees.pkl"
API_VERSION = "1.0.0"
MODEL_NAME = "Extra Trees Classifier"
MODEL_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("churn_api")


app = FastAPI(
    title="API de Predicción de Churn",
    description="API local para predecir abandono de clientes usando un modelo Extra Trees.",
    version=API_VERSION
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    avg_monthly_usage_gb: float = Field(..., ge=0)
    contract_type: Literal["anual", "bianual", "mensual"]
    payment_method: Literal["credito", "debito", "efectivo", "transferencia"]
    internet_service: Literal["cable", "fibra", "movil", "ninguno"]
    has_streaming: int = Field(..., ge=0, le=1)
    has_security_pack: int = Field(..., ge=0, le=1)
    num_products: int = Field(..., ge=0)
    region: Literal["centro", "norte", "oeste", "sur"]
    customer_age: int = Field(..., ge=0)
    is_promo: int = Field(..., ge=0, le=1)


class ChurnResponse(BaseModel):
    prediction: int
    probability_churn: float
    risk_level: Literal["bajo", "medio", "alto"]
    message: str


def load_model():
    try:
        if not MODEL_PATH.exists():
            logger.error("No se encontró el modelo en: %s", MODEL_PATH)
            return None

        loaded_model = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado correctamente desde: %s", MODEL_PATH)

        return loaded_model

    except Exception:
        logger.exception("Error al cargar el modelo")
        return None


model = load_model()

HEALTHCHECK_SAMPLE = {
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


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "alto"
    elif probability >= 0.40:
        return "medio"
    else:
        return "bajo"


def get_message(risk_level: str) -> str:
    if risk_level == "alto":
        return "Cliente con alto riesgo de abandono"
    elif risk_level == "medio":
        return "Cliente con riesgo medio de abandono"
    else:
        return "Cliente con bajo riesgo de abandono"


@app.get("/")
def root():
    return {
        "message": "API de predicción de churn funcionando correctamente",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION
    }

@app.get("/health")
def health():
    if model is None:
        logger.error("Healthcheck fallido: modelo no disponible")

        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    try:
        sample_df = pd.DataFrame([HEALTHCHECK_SAMPLE])

        prediction = int(model.predict(sample_df)[0])
        probability = float(model.predict_proba(sample_df)[:, 1][0])

        return {
            "status": "ok",
            "model_loaded": True,
            "inference_test": True,
            "test_prediction": prediction,
            "test_probability": round(probability, 4),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_file": MODEL_PATH.name,
            "api_version": API_VERSION
        }

    except Exception as exc:
        logger.exception("Falló la prueba de inferencia del healthcheck")

        raise HTTPException(
            status_code=503,
            detail="La API está activa, pero el modelo no supera la prueba de inferencia"
        ) from exc


@app.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    if model is None:
        logger.error("Intento de predicción con el modelo no disponible")

        raise HTTPException(
            status_code=503,
            detail="El modelo de predicción no está disponible"
        )

    try:
        input_data = request.model_dump()
        input_df = pd.DataFrame([input_data])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[:, 1][0])
        probability_rounded = round(probability, 4)

        risk_level = get_risk_level(probability)
        message = get_message(risk_level)

        logger.info(
            "Predicción realizada | prediction=%s | probability=%.4f | risk=%s",
            prediction,
            probability,
            risk_level
        )

        return ChurnResponse(
            prediction=prediction,
            probability_churn=probability_rounded,
            risk_level=risk_level,
            message=message
        )

    except Exception as exc:
        logger.exception("Error durante la inferencia")

        raise HTTPException(
            status_code=500,
            detail="No se pudo generar la predicción"
        ) from exc