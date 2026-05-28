# Proyecto MLOps Local - Predicción de Churn

## 1. Descripción del proyecto

Este proyecto corresponde al trabajo integrador de Laboratorio de Minería de Datos, incluyendo la Etapa 1 de entrenamiento del modelo y la Etapa 2 de despliegue local mediante FastAPI, Docker y Docker Compose.

El objetivo es construir un modelo de Machine Learning capaz de predecir el abandono de clientes, también conocido como churn, para una empresa simulada llamada AndesLink Servicios Digitales S.A.

El problema se aborda como una clasificación binaria:

- 0: el cliente no abandona.
- 1: el cliente abandona.

## 2. Dataset utilizado

Se utilizó el archivo:

data/raw/churn_sintetico.csv

El dataset contiene información relacionada con:

- Antigüedad del cliente.
- Facturación mensual.
- Cargos totales.
- Tickets de soporte.
- Pagos atrasados.
- Uso mensual del servicio.
- Tipo de contrato.
- Método de pago.
- Servicio de internet.
- Servicios adicionales.
- Región.
- Edad del cliente.
- Promociones.
- Variable objetivo churn.

## 3. Flujo de trabajo de la Etapa 1

Durante esta etapa se realizaron los siguientes pasos:

1. Carga del dataset.
2. Análisis exploratorio inicial.
3. Revisión de valores nulos y tipos de datos.
4. Separación de variables predictoras y variable objetivo.
5. Separación en train/test.
6. Preprocesamiento de variables numéricas y categóricas.
7. Entrenamiento de modelos supervisados.
8. Evaluación con métricas de clasificación.
9. Comparación de modelos.
10. Selección del modelo final.
11. Serialización del pipeline entrenado.
12. Registro de experimentos con MLflow.
13. Evidencia de reproducibilidad con DVC.

## 4. Modelos entrenados

Se entrenaron y compararon los siguientes modelos:

- Logistic Regression
- Random Forest
- Decision Tree
- Extra Trees

El modelo seleccionado como artefacto final fue:

Extra Trees Classifier

La elección se justifica porque obtuvo uno de los mejores valores de recall para la clase positiva churn, empatando con Decision Tree, pero con mejor F1-score y mejor ROC-AUC.

## 5. Archivos principales generados

### Modelo final

models/modelo_churn_extra_trees.pkl

Este archivo contiene el pipeline completo:

- Preprocesamiento de variables numéricas.
- Codificación de variables categóricas.
- Modelo Extra Trees entrenado.

### Columnas esperadas por el modelo

models/features_modelo.txt

### Reportes

reports/metricas_modelos.csv  
reports/mlflow_experimentos_resumen.csv  
reports/informe_entrenamiento_etapa1.md  
reports/ejemplo_request_api.json  
reports/ejemplo_response_api.json  
reports/evidencia_reproducibilidad_dvc.txt  

## 6. Instalación de dependencias

Para instalar las dependencias con pip:

pip install -r requirements.txt

Para crear el entorno con conda:

conda env create -f environment.yml  
conda activate churn-mlops

## 7. Ejecución del notebook

El entrenamiento fue desarrollado en Google Colab.

Notebook principal:

entrenamiento_churn_1ipynb.ipynb

El notebook permite reproducir:

- Carga del dataset.
- Entrenamiento de modelos.
- Comparación de métricas.
- Selección del modelo final.
- Serialización del modelo.
- Registro de experimentos con MLflow.
- Evidencia DVC.

## 8. Resultado final

El modelo final quedó guardado como archivo .pkl y fue validado mediante una prueba de carga e inferencia.

Esto permite utilizar el artefacto en la siguiente etapa del proyecto, donde será expuesto mediante una API con FastAPI.



---

## Etapa 2 - Despliegue local con FastAPI y Docker

En la segunda etapa del proyecto se convirtió el modelo entrenado en la Etapa 1 en un servicio de inferencia local.

Para esto se desarrolló una API con FastAPI que carga el modelo serializado, recibe datos de clientes, valida la entrada y devuelve una predicción de churn junto con la probabilidad estimada y un nivel de riesgo.

---

### Arquitectura local

```text
Usuario / Postman
        |
        v
API FastAPI - Endpoint /predict
        |
        v
Pipeline serializado modelo_churn_extra_trees.pkl
        |
        v
Predicción de churn
```
