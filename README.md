# Proyecto MLOps Local - Predicción de Churn

Proyecto integrador de la materia **Laboratorio de Minería de Datos**, orientado a construir un flujo MLOps local para predecir el abandono de clientes, también conocido como **churn**.

El proyecto incluye entrenamiento de modelos, serialización del pipeline, despliegue local con FastAPI, interfaz gráfica con Streamlit, monitoreo técnico con Prometheus y Grafana, pruebas automatizadas y documentación operativa.

---

## 1. Objetivo del proyecto

El objetivo es desarrollar una solución de Machine Learning capaz de predecir si un cliente abandonará o no el servicio de una empresa simulada llamada **AndesLink Servicios Digitales S.A.**

El problema se aborda como una clasificación binaria:

| Clase | Significado |
|---|---|
| 0 | El cliente no abandona |
| 1 | El cliente abandona |

La solución permite obtener:

- predicción de churn
- probabilidad estimada de abandono
- nivel de riesgo
- mensaje interpretativo para el usuario

---

## 2. Dataset utilizado

Se utilizó el archivo:

```text
data/raw/churn_sintetico.csv
```

El dataset contiene variables relacionadas con:

- antigüedad del cliente
- facturación mensual
- cargos totales
- tickets de soporte
- pagos atrasados
- uso mensual del servicio
- tipo de contrato
- método de pago
- servicio de internet
- servicios adicionales
- región
- edad del cliente
- promociones activas
- variable objetivo `churn`

---

## 3. Etapa 1 - Entrenamiento del modelo

Durante la primera etapa se realizó el flujo completo de entrenamiento:

1. Carga del dataset.
2. Análisis exploratorio inicial.
3. Revisión de valores nulos y tipos de datos.
4. Separación de variables predictoras y variable objetivo.
5. División train/test con estratificación.
6. Preprocesamiento con `ColumnTransformer`.
7. Escalado de variables numéricas con `StandardScaler`.
8. Codificación de variables categóricas con `OneHotEncoder`.
9. Entrenamiento de modelos supervisados.
10. Evaluación con métricas de clasificación.
11. Comparación de modelos.
12. Selección del modelo final.
13. Serialización del pipeline entrenado.
14. Registro de experimentos con MLflow.
15. Evidencia de reproducibilidad con DVC.

---

## 4. Modelos entrenados

Se entrenaron y compararon los siguientes modelos:

- Logistic Regression
- Random Forest
- Decision Tree
- Extra Trees

El modelo seleccionado fue:

```text
Extra Trees Classifier
```

La elección se justificó porque obtuvo uno de los mejores valores de recall para la clase positiva `churn`, empatando con Decision Tree, pero con mejor F1-score y mejor ROC-AUC.

---

## 5. Artefactos principales

### Modelo final

```text
models/modelo_churn_extra_trees.pkl
```

Este archivo contiene el pipeline completo:

- preprocesamiento de variables numéricas
- codificación de variables categóricas
- modelo Extra Trees entrenado

### Columnas esperadas por el modelo

```text
models/features_modelo.txt
```

### Reportes

```text
reports/metricas_modelos.csv
reports/mlflow_experimentos_resumen.csv
reports/informe_entrenamiento_etapa1.md
reports/ejemplo_request_api.json
reports/ejemplo_response_api.json
reports/evidencia_reproducibilidad_dvc.txt
```

---

## 6. Etapa 2 - API FastAPI y Docker

En la segunda etapa se convirtió el modelo entrenado en un servicio de inferencia local mediante **FastAPI**.

La API permite:

- cargar el modelo serializado
- validar datos de entrada con Pydantic
- recibir solicitudes JSON
- devolver predicción de churn
- devolver probabilidad de abandono
- clasificar el nivel de riesgo
- exponer documentación automática con Swagger
- exponer métricas para Prometheus

Endpoints principales:

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Verifica que la API esté activa |
| GET | `/health` | Verifica carga del modelo e inferencia real |
| POST | `/predict` | Realiza predicción de churn |
| GET | `/metrics` | Expone métricas para Prometheus |

---

## 7. Etapa 3 - Streamlit y monitoreo

En la tercera etapa se incorporó una interfaz gráfica con **Streamlit** y monitoreo técnico con **Prometheus** y **Grafana**.

El stack local queda compuesto por cuatro servicios:

| Servicio | Puerto | Descripción |
|---|---:|---|
| FastAPI | 8000 | API de predicción |
| Streamlit | 8501 | Interfaz gráfica para usuarios |
| Prometheus | 9090 | Recolección de métricas |
| Grafana | 3000 | Visualización de métricas |

---

## 8. Arquitectura general

```text
Usuario
   |
   v
Streamlit
   |
   v
FastAPI
   |
   v
model_service.py
   |
   v
Pipeline Extra Trees serializado
   |
   v
Predicción de churn
```

Monitoreo:

```text
FastAPI /metrics
   |
   v
Prometheus
   |
   v
Grafana
```

---

## 9. Estructura principal del proyecto

```text
proyecto_churn_mlops/
│
├── data/
│   └── raw/
│
├── docs/
│   ├── index.md
│   ├── arquitectura.md
│   ├── etapa1_entrenamiento.md
│   ├── etapa2_despliegue.md
│   ├── etapa3_monitoreo.md
│   ├── guia_operativa.md
│   └── evidencias.md
│
├── models/
│   ├── modelo_churn_extra_trees.pkl
│   └── features_modelo.txt
│
├── monitoring/
│   └── prometheus.yml
│
├── reports/
│   └── evidencias_etapa3/
│
├── src/
│   ├── api.py
│   ├── model_service.py
│   ├── predict.py
│   ├── streamlit_app.py
│   └── __init__.py
│
├── tests/
│   └── test_api.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── environment.yml
├── mkdocs.yml
└── README.md
```

---

## 10. Instalación de dependencias

Con pip:

```bash
pip install -r requirements.txt
```

Con conda:

```bash
conda env create -f environment.yml
conda activate churn-mlops
```

---

## 11. Ejecución local con Docker Compose

Para construir y levantar el stack completo:

```bash
docker compose up -d --build
```

Para verificar los servicios:

```bash
docker compose ps
```

Resultado esperado:

```text
churn_api          Up ... (healthy)
churn_streamlit    Up ...
prometheus         Up ...
grafana            Up ...
```

URLs locales:

| Servicio | URL |
|---|---|
| FastAPI | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| Streamlit | `http://localhost:8501` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

---

## 12. Uso de la API

### Healthcheck

```bash
http://localhost:8000/health
```

Desde PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Respuesta esperada:

```text
status           : ok
model_loaded     : True
inference_test   : True
model_name       : Extra Trees Classifier
model_version    : 1.0.0
```

### Predicción

Endpoint:

```text
POST /predict
```

Ejemplo de entrada:

```json
{
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
  "is_promo": 1
}
```

Ejemplo de respuesta:

```json
{
  "prediction": 1,
  "probability_churn": 0.7821,
  "risk_level": "alto",
  "message": "Cliente con alto riesgo de abandono"
}
```

---

## 13. Interfaz Streamlit

La interfaz gráfica se encuentra en:

```text
http://localhost:8501
```

Permite completar los datos del cliente, enviar la solicitud a la API y visualizar:

- predicción
- probabilidad de churn
- nivel de riesgo
- mensaje interpretativo
- respuesta JSON técnica

Dentro de Docker, Streamlit se comunica con FastAPI mediante:

```text
http://churn-api:8000
```

Esta configuración se define en `docker-compose.yml`:

```yaml
API_URL: http://churn-api:8000
```

---

## 14. Monitoreo con Prometheus y Grafana

Prometheus recolecta métricas desde:

```text
http://churn-api:8000/metrics
```

Prometheus local:

```text
http://localhost:9090
```

Targets de Prometheus:

```text
http://localhost:9090/targets
```

Grafana local:

```text
http://localhost:3000
```

El dashboard permite visualizar:

- total de requests
- requests por endpoint
- latencia promedio
- predicciones realizadas

---

## 15. Pruebas automatizadas

Para ejecutar la suite de tests:

```powershell
& "C:/Users/Gian/AppData/Local/Programs/Python/Python311/python.exe" -m pytest -v
```

Resultado esperado:

```text
collected 12 items
12 passed
```

La suite cubre:

- endpoint raíz
- healthcheck
- predicción válida
- campos faltantes
- tipos incorrectos
- valores fuera de rango
- categorías inválidas
- variables binarias inválidas
- modelo no disponible
- fallas de inferencia

---

## 16. Documentación del proyecto

La documentación está construida con MkDocs.

Para generar la documentación local:

```bash
python -m mkdocs build
```

Para servirla localmente:

```bash
python -m mkdocs serve
```

Archivos principales de documentación:

```text
docs/index.md
docs/arquitectura.md
docs/etapa1_entrenamiento.md
docs/etapa2_despliegue.md
docs/etapa3_monitoreo.md
docs/guia_operativa.md
docs/evidencias.md
```

---

## 17. Evidencias

El proyecto incluye evidencias de:

- ejecución de Docker Compose
- healthcheck de API
- pruebas con Postman
- pruebas automatizadas con pytest
- Prometheus targets
- dashboard de Grafana
- interfaz Streamlit con predicción real

Ejemplo de evidencia reciente:

```text
reports/evidencias_etapa3/streamlit_docker_prediccion_alto_riesgo.png
```

---

## 18. Detener servicios

Para detener los contenedores sin borrar datos:

```bash
docker compose stop
```

Para volver a levantarlos:

```bash
docker compose up -d
```

Para detener y eliminar contenedores:

```bash
docker compose down
```

No usar `docker compose down -v` salvo que se quiera borrar también el volumen de Grafana.

---

## 19. Estado actual del proyecto

El proyecto cuenta actualmente con:

- modelo entrenado y serializado
- API FastAPI funcional
- validación robusta con Pydantic
- manejo de errores con HTTPException
- healthcheck con inferencia real
- logging
- métricas Prometheus
- dashboard Grafana
- interfaz Streamlit
- Docker Compose con cuatro servicios
- 12 tests automatizados
- documentación operativa con MkDocs
- evidencias del funcionamiento

---

## 20. Autor

**Gianmarco Rios**  
Tecnicatura en Ciencia de Datos e Inteligencia Artificial  
Laboratorio de Minería de Datos