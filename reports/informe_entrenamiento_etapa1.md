
# Informe técnico - Etapa 1: Entrenamiento del modelo de churn

## 1. Contexto del problema

AndesLink Servicios Digitales S.A. es una empresa simulada de suscripción mensual que necesita anticipar el abandono de clientes.  
El objetivo del proyecto es construir un modelo de Machine Learning capaz de predecir si un cliente tiene riesgo de abandonar el servicio.

El problema se formula como una tarea de clasificación binaria:

- `0`: el cliente no abandona.
- `1`: el cliente abandona.

Desde el punto de vista del negocio, detectar clientes con riesgo de churn permite activar campañas de retención antes de perderlos.

---

## 2. Descripción del dataset

Se utilizó el archivo `churn_sintetico.csv`.

El dataset contiene:

- Cantidad de registros: 5000
- Cantidad de columnas: 16
- Variable objetivo: `churn`

Distribución de la variable objetivo:

- Clientes que no abandonan: 3298 (65.96%)
- Clientes que abandonan: 1702 (34.04%)

Las variables utilizadas incluyen información relacionada con antigüedad del cliente, facturación, uso mensual del servicio, pagos atrasados, tickets de soporte, tipo de contrato, método de pago, región y servicios contratados.

---

## 3. Preparación de datos

Se separaron las variables predictoras de la variable objetivo.

Las variables numéricas fueron transformadas con `StandardScaler`, mientras que las variables categóricas fueron codificadas mediante `OneHotEncoder`.

Para organizar este flujo se utilizó un `ColumnTransformer`, integrado luego dentro de un `Pipeline` de scikit-learn.

Esto permite guardar en un único artefacto tanto el preprocesamiento como el modelo final, facilitando su uso posterior en una API de inferencia.

---

## 4. Separación train/test

El dataset fue dividido en conjunto de entrenamiento y conjunto de prueba usando `train_test_split`.

Se utilizó `stratify=y` para mantener la proporción de clientes que abandonan y no abandonan en ambos conjuntos.

Esto permite realizar una evaluación más representativa del rendimiento del modelo.

---

## 5. Modelos entrenados

Se entrenaron cuatro modelos supervisados de clasificación binaria:

1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Extra Trees

Todos los modelos fueron integrados en pipelines que incluyen el preprocesamiento y el algoritmo de clasificación.

---

## 6. Comparación de métricas

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Extra Trees | 0.6670 | 0.5075 | 0.6971 | 0.5874 | 0.7338 |
| Decision Tree | 0.6470 | 0.4867 | 0.6971 | 0.5732 | 0.7046 |
| Random Forest | 0.6890 | 0.5367 | 0.6235 | 0.5769 | 0.7354 |
| Logistic Regression | 0.7230 | 0.6207 | 0.4765 | 0.5391 | 0.7577 |


---

## 7. Análisis de resultados

Logistic Regression obtuvo el mejor accuracy y el mejor ROC-AUC, pero presentó un recall bajo para la clase positiva churn.  
Esto significa que, aunque acierta bien en términos generales, deja sin detectar una cantidad importante de clientes que realmente abandonan.

Random Forest mejoró el recall respecto de Logistic Regression, logrando detectar una mayor proporción de clientes con riesgo de churn.

Decision Tree obtuvo uno de los mejores valores de recall, pero presentó menor precision y mayor cantidad de falsos positivos.

Extra Trees igualó a Decision Tree en recall, pero obtuvo mejor F1-score y mejor ROC-AUC, logrando un equilibrio más sólido entre detección de churn y rendimiento general.

---

## 8. Modelo final seleccionado

Se seleccionó como modelo final inicial el modelo `Extra Trees Classifier`.

La elección se justifica porque obtuvo uno de los mejores valores de recall para la clase positiva, lo cual es relevante para el caso de negocio.  
En problemas de churn, es importante detectar la mayor cantidad posible de clientes que pueden abandonar el servicio, ya que esto permite tomar acciones preventivas.

Aunque Logistic Regression obtuvo mayor accuracy, Extra Trees resulta más adecuado para este problema porque prioriza la detección de clientes en riesgo.

---

## 9. Serialización del modelo

El pipeline final fue serializado utilizando `joblib` en formato `.pkl`.

Archivo generado:

`models/modelo_churn_extra_trees.pkl`

Este archivo incluye:

- Preprocesamiento de variables numéricas.
- Codificación de variables categóricas.
- Modelo Extra Trees entrenado.

Luego se validó que el modelo pudiera cargarse correctamente desde disco y realizar una predicción sobre un cliente del conjunto de prueba.

---

## 10. Ejemplo de inferencia

Se probó el modelo cargado con un cliente del conjunto de prueba.

El modelo devolvió:

- Predicción: 1
- Probabilidad de churn: 0.7821

Esto demuestra que el artefacto serializado puede utilizarse para inferencia sin necesidad de reentrenar el modelo.

---

## 11. Archivos generados

Durante esta etapa se generaron los siguientes archivos relevantes:

- `models/modelo_churn_extra_trees.pkl`
- `models/features_modelo.txt`
- `reports/metricas_modelos.csv`
- `reports/ejemplo_request_api.json`
- `reports/ejemplo_response_api.json`

---

## 12. Conclusión

En esta primera etapa se construyó un flujo completo de entrenamiento para un modelo de predicción de churn.  
Se realizó análisis inicial del dataset, preparación de datos, entrenamiento de modelos, comparación mediante métricas, selección del modelo final y serialización del pipeline.

El modelo final queda preparado para ser utilizado en la siguiente etapa del proyecto, donde será expuesto mediante una API de inferencia con FastAPI.
