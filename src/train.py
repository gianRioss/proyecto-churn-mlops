from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "churn_sintetico.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_churn_extra_trees.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features_modelo.txt"

METRICS_PATH = BASE_DIR / "reports" / "metricas_modelos.csv"
REQUEST_EXAMPLE_PATH = BASE_DIR / "reports" / "ejemplo_request_api.json"
RESPONSE_EXAMPLE_PATH = BASE_DIR / "reports" / "ejemplo_response_api.json"

TARGET_COLUMN = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


EXAMPLE_REQUEST = {
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


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_pipeline(preprocessor, model):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_models(preprocessor):
    return {
        "Extra Trees": build_pipeline(
            preprocessor,
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Decision Tree": build_pipeline(
            preprocessor,
            DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": build_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Logistic Regression": build_pipeline(
            preprocessor,
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def get_risk_level(probability):
    if probability >= 0.70:
        return "alto"

    if probability >= 0.40:
        return "medio"

    return "bajo"


def get_message(risk_level):
    if risk_level == "alto":
        return "Cliente con alto riesgo de abandono"

    if risk_level == "medio":
        return "Cliente con riesgo medio de abandono"

    return "Cliente con bajo riesgo de abandono"


def save_features(feature_names):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        for feature in feature_names:
            file.write(f"{feature}\n")


def save_api_examples(final_model):
    example_df = pd.DataFrame([EXAMPLE_REQUEST])

    prediction = int(final_model.predict(example_df)[0])
    probability = float(final_model.predict_proba(example_df)[:, 1][0])
    risk_level = get_risk_level(probability)

    response = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "message": get_message(risk_level),
    }

    REQUEST_EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REQUEST_EXAMPLE_PATH, "w", encoding="utf-8") as file:
        json.dump(EXAMPLE_REQUEST, file, indent=2, ensure_ascii=False)

    with open(RESPONSE_EXAMPLE_PATH, "w", encoding="utf-8") as file:
        json.dump(response, file, indent=2, ensure_ascii=False)


def main():
    print("Cargando dataset...")
    dataframe = pd.read_csv(DATA_PATH)

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Registros: {len(dataframe)}")
    print(f"Variables predictoras: {len(X.columns)}")
    print(f"Columnas numericas: {numeric_features}")
    print(f"Columnas categoricas: {categorical_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    results = []

    for model_name, model in models.items():
        print(f"Entrenando {model_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["modelo"] = model_name
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[
        ["modelo", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    final_model_name = "Extra Trees"
    final_model = models[final_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    save_features(X.columns.tolist())
    save_api_examples(final_model)

    print("Entrenamiento finalizado correctamente.")
    print(f"Modelo final seleccionado: {final_model_name}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()