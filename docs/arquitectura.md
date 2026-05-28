# Arquitectura del proyecto

## Visión general

El proyecto implementa una arquitectura local de MLOps para entrenar, versionar, desplegar y probar un modelo de predicción de churn.

La solución está organizada en etapas:

- **Etapa 1:** entrenamiento, evaluación y serialización del modelo.
- **Etapa 2:** despliegue local mediante una API de inferencia.
- **Etapa 3:** monitoreo técnico y monitoreo de datos.

---

## Arquitectura general

```text
Dataset CSV
    |
    v
EDA y preparación de datos
    |
    v
Pipeline de preprocesamiento + modelo
    |
    v
Modelo serializado .pkl
    |
    v
API FastAPI
    |
    v
Cliente / Postman