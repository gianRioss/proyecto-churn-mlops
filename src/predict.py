import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "modelo_churn_extra_trees.pkl"
REQUEST_PATH = BASE_DIR / "reports" / "ejemplo_request_api.json"


def load_model():
    """Carga el modelo serializado desde la carpeta models."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


def load_sample_request():
    """Carga un ejemplo JSON de cliente para probar inferencia."""
    if not REQUEST_PATH.exists():
        raise FileNotFoundError(f"No se encontró el JSON de prueba en: {REQUEST_PATH}")

    with open(REQUEST_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def predict_churn(model, input_data):
    """Realiza una predicción de churn usando el modelo cargado."""
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0]

    return int(prediction), float(probability)


def get_risk_level(probability):
    """Convierte la probabilidad en una categoría de riesgo."""
    if probability >= 0.70:
        return "alto"
    elif probability >= 0.40:
        return "medio"
    else:
        return "bajo"


if __name__ == "__main__":
    print("Cargando modelo...")
    model = load_model()
    print("Modelo cargado correctamente.")

    print("\nCargando ejemplo de request...")
    sample_request = load_sample_request()
    print("Request cargado correctamente.")

    prediction, probability = predict_churn(model, sample_request)
    risk_level = get_risk_level(probability)

    print("\nResultado de inferencia")
    print("----------------------")
    print(f"Predicción: {prediction}")
    print(f"Probabilidad de churn: {round(probability, 4)}")
    print(f"Nivel de riesgo: {risk_level}")