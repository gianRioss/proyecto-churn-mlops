import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "modelo_churn_extra_trees.pkl"

MODEL_NAME = "Extra Trees Classifier"
MODEL_VERSION = "1.0.0"


logger = logging.getLogger("churn_api")


def load_model() -> Any | None:
    """Carga el modelo serializado desde la carpeta models."""
    try:
        if not MODEL_PATH.exists():
            logger.error(
                "No se encontró el modelo en: %s",
                MODEL_PATH
            )
            return None

        loaded_model = joblib.load(MODEL_PATH)

        logger.info(
            "Modelo cargado correctamente desde: %s",
            MODEL_PATH
        )

        return loaded_model

    except Exception:
        logger.exception("Error al cargar el modelo")
        return None


def predict_with_model(
    model: Any,
    input_data: dict
) -> tuple[int, float]:
    """Realiza una predicción y devuelve clase y probabilidad."""
    input_df = pd.DataFrame([input_data])

    prediction = int(model.predict(input_df)[0])

    probability = float(
        model.predict_proba(input_df)[:, 1][0]
    )

    return prediction, probability


def get_risk_level(probability: float) -> str:
    """Clasifica la probabilidad de churn por nivel de riesgo."""
    if probability >= 0.70:
        return "alto"

    if probability >= 0.40:
        return "medio"

    return "bajo"


def get_message(risk_level: str) -> str:
    """Genera el mensaje correspondiente al nivel de riesgo."""
    if risk_level == "alto":
        return "Cliente con alto riesgo de abandono"

    if risk_level == "medio":
        return "Cliente con riesgo medio de abandono"

    return "Cliente con bajo riesgo de abandono"