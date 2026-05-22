from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_churn_extra_trees.pkl"


app = FastAPI(
    title="API de Predicción de Churn",
    description="API local para predecir abandono de clientes usando un modelo Extra Trees.",
    version="1.0.0"
)


class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    avg_monthly_usage_gb: float = Field(..., ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    has_streaming: int = Field(..., ge=0, le=1)
    has_security_pack: int = Field(..., ge=0, le=1)
    num_products: int = Field(..., ge=0)
    region: str
    customer_age: int = Field(..., ge=0)
    is_promo: int = Field(..., ge=0, le=1)


class ChurnResponse(BaseModel):
    prediction: int
    probability_churn: float
    risk_level: Literal["bajo", "medio", "alto"]
    message: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


model = load_model()


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
        "model": "Extra Trees Classifier",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    input_data = request.model_dump()
    input_df = pd.DataFrame([input_data])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    probability_rounded = round(probability, 4)

    risk_level = get_risk_level(probability)
    message = get_message(risk_level)

    return ChurnResponse(
        prediction=prediction,
        probability_churn=probability_rounded,
        risk_level=risk_level,
        message=message
    )