from pathlib import Path
import json

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "churn_sintetico.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "evidently"

HTML_REPORT_PATH = OUTPUT_DIR / "data_drift_report.html"
JSON_SUMMARY_PATH = OUTPUT_DIR / "data_drift_summary.json"

TARGET_COLUMN = "churn"
RANDOM_STATE = 42


def load_reference_data():
    dataframe = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN in dataframe.columns:
        dataframe = dataframe.drop(columns=[TARGET_COLUMN])

    return dataframe


def create_current_data(reference_data):
    current_data = reference_data.sample(frac=0.30, random_state=RANDOM_STATE).copy()

    current_data["monthly_charge"] = current_data["monthly_charge"] * 1.20
    current_data["support_tickets"] = current_data["support_tickets"] + 1
    current_data["avg_monthly_usage_gb"] = current_data["avg_monthly_usage_gb"] * 1.15

    current_data.loc[current_data.sample(frac=0.35, random_state=RANDOM_STATE).index, "contract_type"] = "mensual"
    current_data.loc[current_data.sample(frac=0.25, random_state=24).index, "payment_method"] = "credito"

    return current_data


def save_json_summary(report, reference_data, current_data):
    report_dict = report.as_dict()

    summary = {
        "report_name": "Data Drift Report",
        "tool": "Evidently",
        "evidently_version": "0.6.7",
        "reference_rows": len(reference_data),
        "current_rows": len(current_data),
        "reference_columns": len(reference_data.columns),
        "current_columns": len(current_data.columns),
        "simulated_changes": [
            "monthly_charge aumentado un 20%",
            "support_tickets aumentado en 1",
            "avg_monthly_usage_gb aumentado un 15%",
            "mayor proporción de contract_type mensual",
            "mayor proporción de payment_method credito",
        ],
        "report_file": str(HTML_REPORT_PATH.relative_to(BASE_DIR)),
        "raw_report": report_dict,
    }

    with open(JSON_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos de referencia...")
    reference_data = load_reference_data()

    print("Generando datos actuales simulados...")
    current_data = create_current_data(reference_data)

    print("Generando reporte de data drift con Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report.save_html(str(HTML_REPORT_PATH))
    save_json_summary(report, reference_data, current_data)

    print("Reporte de data drift generado correctamente.")
    print(f"HTML: {HTML_REPORT_PATH}")
    print(f"JSON: {JSON_SUMMARY_PATH}")


if __name__ == "__main__":
    main()