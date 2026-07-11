# Etapa 3 - Monitoreo y mejora continua

## Objetivo de la etapa

La Etapa 3 incorpora componentes de monitoreo técnico, visualización, pruebas operativas y seguimiento de calidad de datos para el modelo de predicción de churn.

El objetivo es acercar el proyecto a un flujo MLOps local más completo, donde no solo se entrena y despliega un modelo, sino que también se puede observar su funcionamiento, probar la API, detectar posibles cambios en los datos y reproducir el entrenamiento.

---

## Componentes implementados

Durante esta etapa se incorporaron los siguientes componentes:

- monitoreo técnico con Prometheus;
- dashboard de visualización con Grafana;
- interfaz gráfica con Streamlit;
- colección Postman reutilizable;
- pipeline reproducible con DVC;
- reporte de data drift con Evidently;
- pruebas automatizadas ampliadas con pytest;
- documentación operativa del stack.

---

## Monitoreo técnico con Prometheus

Se agregó el endpoint `/metrics` en la API FastAPI mediante `prometheus-fastapi-instrumentator`.

Prometheus consulta periódicamente las métricas de la API desde el contenedor `churn-api`.

Archivo de configuración:

```text
monitoring/prometheus.yml