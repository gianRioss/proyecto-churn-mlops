import json
from pathlib import Path

from src.model_service import (
    get_risk_level,
    load_model,
    predict_with_model,
)


BASE_DIR = Path(__file__).resolve().parent.parent

REQUEST_PATH = (
    BASE_DIR
    / "reports"
    / "ejemplo_request_api.json"
)


def load_sample_request() -> dict:
    """Carga un ejemplo JSON para probar una inferencia local."""
    if not REQUEST_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el JSON de prueba en: {REQUEST_PATH}"
        )

    with open(
        REQUEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main() -> None:
    """Ejecuta una predicción local usando el ejemplo de request."""
    print("Cargando modelo...")

    model = load_model()

    if model is None:
        raise RuntimeError(
            "No fue posible cargar el modelo de predicción."
        )

    print("Modelo cargado correctamente.")

    print("\nCargando ejemplo de request...")

    sample_request = load_sample_request()

    print("Request cargado correctamente.")

    prediction, probability = predict_with_model(
        model,
        sample_request
    )

    risk_level = get_risk_level(probability)

    print("\nResultado de inferencia")
    print("----------------------")
    print(f"Predicción: {prediction}")
    print(
        "Probabilidad de churn: "
        f"{round(probability, 4)}"
    )
    print(f"Nivel de riesgo: {risk_level}")


if __name__ == "__main__":
    main()