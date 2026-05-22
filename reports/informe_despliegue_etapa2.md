# Informe técnico - Etapa 2: Despliegue local del modelo de churn

## 1. Objetivo de la etapa

El objetivo de esta segunda etapa fue convertir el modelo entrenado en la Etapa 1 en un servicio de inferencia utilizable localmente.

Para esto se desarrolló una API con FastAPI, capaz de cargar el modelo serializado, recibir datos de clientes mediante un endpoint de predicción y devolver como respuesta la predicción de churn, la probabilidad estimada y un nivel de riesgo.

---

## 2. Arquitectura local de la solución

La arquitectura implementada es local y está compuesta por los siguientes componentes:

```text
Usuario / Postman
        |
        v
FastAPI - Endpoint /predict
        |
        v
Pipeline serializado .pkl
        |
        v
Predicción de churn