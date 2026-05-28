# Etapa 1 - Entrenamiento del modelo

## Objetivo de la etapa

El objetivo de la primera etapa fue construir un modelo supervisado de clasificación binaria capaz de predecir si un cliente abandonará o no el servicio.

El problema se definió como un caso de **churn prediction**, donde la variable objetivo indica:

- `0`: el cliente no abandona.
- `1`: el cliente abandona.

Desde el punto de vista de negocio, detectar clientes con riesgo de abandono permite anticipar acciones de retención y mejorar la toma de decisiones.

---

## Dataset utilizado

Se utilizó el archivo:

```text
data/raw/churn_sintetico.csv