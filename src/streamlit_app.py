import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


st.set_page_config(
    page_title="Predicción de Churn",
    page_icon="📊",
    layout="wide"
)

def check_api_health() -> tuple[bool, str]:
    """Verifica si la API y el modelo están disponibles."""

    try:
        response = httpx.get(
            f"{API_BASE_URL}/health",
            timeout=5.0
        )

        if response.status_code == 200:
            return True, "API y modelo disponibles"

        return (
            False,
            f"La API respondió con código {response.status_code}"
        )

    except httpx.RequestError:
        return False, "No se pudo conectar con la API"


def display_prediction_result(result):
    """Muestra visualmente el resultado de la predicción."""

    prediction = result["prediction"]
    probability = float(result["probability_churn"])
    risk_level = result["risk_level"]
    message = result["message"]

    prediction_label = (
        "Abandona"
        if prediction == 1
        else "Permanece"
    )

    st.divider()
    st.subheader("Resultado del análisis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicción",
            prediction_label
        )

    with col2:
        st.metric(
            "Probabilidad de churn",
            f"{probability:.1%}"
        )

    with col3:
        st.metric(
            "Nivel de riesgo",
            risk_level.capitalize()
        )

    st.write("Probabilidad estimada de abandono:")

    st.progress(probability)

    if risk_level == "alto":
        st.error(message)

    elif risk_level == "medio":
        st.warning(message)

    else:
        st.success(message)

    with st.expander("Ver respuesta JSON de la API"):
        st.json(result)
    
    
st.title("Predicción de Churn de Clientes")

st.write(
    "Complete los datos del cliente para estimar "
    "su probabilidad de abandono."
)


api_available, api_status = check_api_health()


with st.sidebar:
    st.header("Estado del sistema")

    if api_available:
        st.success(api_status)
    else:
        st.error(api_status)

    st.caption(f"API: {API_BASE_URL}")
    
with st.form("churn_prediction_form"):
    st.subheader("Datos del cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure_months = st.number_input(
            "Antigüedad en meses",
            min_value=0,
            value=21,
            step=1
        )

        monthly_charge = st.number_input(
            "Cargo mensual",
            min_value=0.0,
            value=62.45,
            step=0.01,
            format="%.2f"
        )

        total_charges = st.number_input(
            "Cargos totales",
            min_value=0.0,
            value=1365.15,
            step=0.01,
            format="%.2f"
        )

        support_tickets = st.number_input(
            "Tickets de soporte",
            min_value=0,
            value=1,
            step=1
        )

        late_payments = st.number_input(
            "Pagos atrasados",
            min_value=0,
            value=1,
            step=1
        )
    with col2:
        avg_monthly_usage_gb = st.number_input(
            "Uso mensual promedio (GB)",
            min_value=0.0,
            value=142.87,
            step=0.01,
            format="%.2f"
        )

        contract_type = st.selectbox(
            "Tipo de contrato",
            options=[
                "mensual",
                "anual",
                "bianual"
            ],
            format_func=str.capitalize
        )

        payment_method = st.selectbox(
            "Método de pago",
            options=[
                "credito",
                "debito",
                "efectivo",
                "transferencia"
            ],
            format_func=str.capitalize
        )

        internet_service = st.selectbox(
            "Servicio de internet",
            options=[
                "movil",
                "cable",
                "fibra",
                "ninguno"
            ],
            format_func=str.capitalize
        )

        region = st.selectbox(
            "Región",
            options=[
                "norte",
                "sur",
                "centro",
                "oeste"
            ],
            format_func=str.capitalize
        )
    with col3:
        has_streaming = st.selectbox(
            "¿Tiene streaming?",
            options=[1, 0],
            format_func=lambda value: (
                "Sí" if value == 1 else "No"
            )
        )

        has_security_pack = st.selectbox(
            "¿Tiene paquete de seguridad?",
            options=[1, 0],
            format_func=lambda value: (
                "Sí" if value == 1 else "No"
            )
        )

        num_products = st.number_input(
            "Cantidad de productos",
            min_value=0,
            value=3,
            step=1
        )

        customer_age = st.number_input(
            "Edad del cliente",
            min_value=0,
            value=52,
            step=1
        )

        is_promo = st.selectbox(
            "¿Tiene promoción activa?",
            options=[1, 0],
            format_func=lambda value: (
                "Sí" if value == 1 else "No"
            )
        ) 
    submitted = st.form_submit_button(
        "Analizar riesgo de churn",
        type="primary",
        use_container_width=True
    )


if submitted:
    payload = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "has_streaming": has_streaming,
        "has_security_pack": has_security_pack,
        "num_products": num_products,
        "region": region,
        "customer_age": customer_age,
        "is_promo": is_promo
    }

    try:
        with st.spinner("Analizando el riesgo de abandono..."):
            response = httpx.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=10.0
            )

        if response.status_code == 200:
            result = response.json()

            display_prediction_result(result)

        else:
            st.error(
                f"La API respondió con código "
                f"{response.status_code}"
            )

    except httpx.RequestError:
        st.error(
            "No se pudo conectar con la API. "
            "Verifique que FastAPI esté funcionando."
        )