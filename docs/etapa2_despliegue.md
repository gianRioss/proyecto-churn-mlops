# Etapa 2 - Despliegue local con FastAPI y Docker

## Objetivo de la etapa

El objetivo de la segunda etapa fue convertir el modelo entrenado en la Etapa 1 en un servicio de inferencia local.

Para esto se desarrolló una API con FastAPI, capaz de cargar el modelo serializado, recibir datos de clientes, validar la entrada y devolver una predicción de churn.

---

## API de inferencia

La API se implementó en el archivo:

```text
src/api.py