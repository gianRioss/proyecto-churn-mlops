# Proyecto MLOps Local - Predicción de Churn

## Resumen del proyecto

Este proyecto corresponde al trabajo integrador de la materia **Laboratorio de Minería de Datos**.

El objetivo es construir una solución local de MLOps para predecir el abandono de clientes, también conocido como **churn**, para una empresa simulada llamada **AndesLink Servicios Digitales S.A.**

La solución cubre el ciclo completo de un proyecto de Machine Learning aplicado:

- comprensión del problema de negocio;
- análisis exploratorio de datos;
- entrenamiento y evaluación de modelos;
- serialización del modelo final;
- trazabilidad con MLflow;
- versionado de datos/modelo con DVC;
- despliegue local mediante FastAPI;
- pruebas con Postman y pytest;
- contenedorización con Docker y Docker Compose;
- preparación para monitoreo técnico y de datos en la Etapa 3.

---

## Estado actual del proyecto

Actualmente se completaron las siguientes etapas:

### Etapa 1 - Entrenamiento

Se entrenaron y compararon modelos de clasificación binaria para predecir churn.  
El modelo seleccionado fue **Extra Trees Classifier**, priorizando el recall para detectar clientes con riesgo de abandono.

### Etapa 2 - Despliegue local

Se desarrolló una API con **FastAPI** que carga el modelo serializado y permite obtener predicciones mediante el endpoint `/predict`.

Además, se agregaron pruebas automáticas con **pytest** y despliegue reproducible mediante **Docker Compose**.

### Etapa 3 - Monitoreo

La próxima etapa incorporará monitoreo técnico con **Prometheus** y **Grafana**, además de monitoreo de datos o drift con **Evidently**.

---

## Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Pydantic
- Pytest
- Docker
- Docker Compose
- MkDocs Material
- Git y GitHub