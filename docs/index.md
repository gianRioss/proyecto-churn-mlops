# Proyecto MLOps Local - Predicción de Churn

## Resumen del proyecto

Este proyecto corresponde al trabajo integrador de la materia **Laboratorio de Minería de Datos**.

El objetivo es construir una solución local de MLOps para predecir el abandono de clientes, también conocido como **churn**, para una empresa simulada llamada **AndesLink Servicios Digitales S.A.**

La solución cubre el ciclo completo de un proyecto de Machine Learning aplicado:

- comprensión del problema de negocio;
- análisis exploratorio de datos;
- entrenamiento y evaluación de modelos;
- serialización del modelo final;
- trazabilidad inicial con MLflow;
- entrenamiento reproducible con DVC;
- despliegue local mediante FastAPI;
- validación de datos de entrada con Pydantic;
- pruebas con pytest y Postman;
- interfaz gráfica con Streamlit;
- contenedorización con Docker y Docker Compose;
- monitoreo técnico con Prometheus y Grafana;
- análisis de data drift con Evidently;
- documentación operativa con MkDocs.

---

## Estado actual del proyecto

Actualmente el proyecto cuenta con un stack MLOps local completo para entrenamiento, despliegue, pruebas, monitoreo y documentación.

### Etapa 1 - Entrenamiento

Se entrenaron y compararon modelos de clasificación binaria para predecir churn.

Modelos evaluados:

- Logistic Regression;
- Random Forest;
- Decision Tree;
- Extra Trees.

El modelo seleccionado fue **Extra Trees Classifier**, priorizando la detección de clientes con riesgo de abandono.

Artefactos principales:

```text
models/modelo_churn_extra_trees.pkl
models/features_modelo.txt
reports/metricas_modelos.csv
reports/informe_entrenamiento_etapa1.md