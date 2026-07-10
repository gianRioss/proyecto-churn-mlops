# Guía operativa del proyecto

Esta guía explica cómo levantar, verificar, probar y detener el entorno local del proyecto MLOps de predicción de churn.

El sistema se ejecuta con Docker Compose e incluye cuatro servicios principales:

| Servicio | Descripción | URL local |
|---|---|---|
| FastAPI | API de predicción de churn | `http://localhost:8000` |
| Streamlit | Interfaz gráfica para usuarios | `http://localhost:8501` |
| Prometheus | Recolección de métricas | `http://localhost:9090` |
| Grafana | Visualización de métricas | `http://localhost:3000` |

---

## 1. Requisitos previos

Antes de ejecutar el proyecto, se necesita tener instalado:

- Docker Desktop
- Git
- Python 3.11
- Navegador web

Los comandos deben ejecutarse desde la raíz del proyecto:

```bash
proyecto_churn_mlops/
```

---

## 2. Levantar el stack completo

Para construir y levantar todos los servicios:

```bash
docker compose up -d --build
```

Este comando construye las imágenes necesarias y levanta:

- API FastAPI
- Interfaz Streamlit
- Prometheus
- Grafana

---

## 3. Verificar contenedores

Para revisar el estado de los servicios:

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

El servicio `churn_api` debe figurar como `healthy`.

---

## 4. Probar la API

Healthcheck:

```text
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

Swagger / documentación automática:

```text
http://localhost:8000/docs
```

---

## 5. Probar Streamlit

Abrir:

```text
http://localhost:8501
```

La interfaz permite completar los datos del cliente y enviar la solicitud a la API.

Flujo:

```text
Usuario
   ↓
Streamlit
   ↓
POST /predict
   ↓
FastAPI
   ↓
Modelo Extra Trees
   ↓
Resultado visual
```

Dentro de Docker, Streamlit se comunica con la API usando:

```text
http://churn-api:8000
```

Esto se define en `docker-compose.yml` mediante:

```yaml
API_URL: http://churn-api:8000
```

Un resultado esperado con los valores por defecto es:

```text
Predicción: Abandona
Probabilidad de churn: 78.2%
Nivel de riesgo: Alto
Mensaje: Cliente con alto riesgo de abandono
```

---

## 6. Verificar Prometheus

Abrir:

```text
http://localhost:9090/targets
```

El target de la API debe aparecer como:

```text
UP
```

Prometheus recolecta métricas desde:

```text
http://churn-api:8000/metrics
```

---

## 7. Verificar Grafana

Abrir:

```text
http://localhost:3000
```

Grafana permite visualizar métricas como:

- cantidad total de requests
- requests por endpoint
- latencia promedio
- predicciones realizadas

La fuente de datos debe apuntar internamente a:

```text
http://prometheus:9090
```

---

## 8. Ejecutar pruebas automatizadas

Para ejecutar la suite de tests:

```powershell
& "C:/Users/Gian/AppData/Local/Programs/Python/Python311/python.exe" -m pytest -v
```

Resultado esperado:

```text
collected 12 items
12 passed
```

Las pruebas cubren:

- endpoint raíz
- healthcheck
- predicción válida
- campo faltante
- tipo incorrecto
- valor fuera de rango
- categoría inválida
- variable binaria inválida
- modelo no disponible
- falla de inferencia

---

## 9. Detener servicios

Para detener los contenedores sin borrar datos:

```bash
docker compose stop
```

Para volver a levantarlos:

```bash
docker compose up -d
```

---

## 10. Eliminar contenedores

Para detener y eliminar los contenedores:

```bash
docker compose down
```

Este comando no elimina el volumen de Grafana.

Para eliminar también volúmenes:

```bash
docker compose down -v
```

Usar `-v` solamente si se desea borrar la configuración y datos guardados de Grafana.

---

## 11. Troubleshooting

### La API no aparece como healthy

Revisar logs:

```bash
docker compose logs churn-api
```

Verificar estado:

```bash
docker compose ps
```

La API debe aparecer como:

```text
Up ... (healthy)
```

---

### Streamlit no se conecta con la API

Verificar que en `docker-compose.yml` exista:

```yaml
API_URL: http://churn-api:8000
```

Dentro de Docker no se debe usar:

```text
http://127.0.0.1:8000
```

porque `127.0.0.1` apuntaría al propio contenedor de Streamlit, no al contenedor de la API.

---

### Prometheus no encuentra la API

Revisar:

```text
monitoring/prometheus.yml
```

Debe apuntar a:

```yaml
targets: ["churn-api:8000"]
```

---

### Grafana no muestra métricas

Verificar primero Prometheus:

```text
http://localhost:9090/targets
```

Luego revisar que la fuente de datos de Grafana apunte a:

```text
http://prometheus:9090
```

---

## 12. Comandos principales

| Acción | Comando |
|---|---|
| Construir y levantar todo | `docker compose up -d --build` |
| Ver contenedores | `docker compose ps` |
| Ver logs de API | `docker compose logs churn-api` |
| Ver logs de Streamlit | `docker compose logs streamlit` |
| Detener servicios | `docker compose stop` |
| Eliminar contenedores | `docker compose down` |
| Ejecutar tests | `python -m pytest -v` |
