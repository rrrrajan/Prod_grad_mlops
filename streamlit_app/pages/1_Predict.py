import streamlit as st
import requests
from config import PREDICT_ENDPOINT
import os

# -------------------------------------------------------
# API Configuration
# -------------------------------------------------------

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

API_PREFIX = "/api/v1"

PREDICT_ENDPOINT = f"{API_BASE_URL}{API_PREFIX}/predict"


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.markdown(
    """
Predict whether a telecom customer is likely to churn based on
their demographic information and subscribed services.
"""
)

st.divider()

# -------------------------------------------------------
# Prediction Form
# -------------------------------------------------------

with st.form("prediction_form"):

    # =====================================================
    # Customer Information
    # =====================================================

    st.subheader("👤 Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

    with col2:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12,
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=850.0,
            step=10.0,
        )

    st.divider()

    # =====================================================
    # Phone Services
    # =====================================================

    st.subheader("☎️ Phone Services")

    col1, col2 = st.columns(2)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

    with col2:
        if phone_service == "No":
            multiple_line_options = ["No phone service"]
        else:
            multiple_line_options = ["No", "Yes"]

        multiple_lines = st.selectbox(
            "Multiple Lines",
            multiple_line_options,
        )

    st.divider()

    # =====================================================
    # Internet Services
    # =====================================================

    st.subheader("🌐 Internet Services")

    col1, col2 = st.columns(2)

    with col1:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        if internet_service == "No":
            internet_options = ["No internet service"]
        else:
            internet_options = ["No", "Yes"]

        online_security = st.selectbox(
            "Online Security",
            internet_options,
        )

        online_backup = st.selectbox(
            "Online Backup",
            internet_options,
        )

        device_protection = st.selectbox(
            "Device Protection",
            internet_options,
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            internet_options,
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            internet_options,
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            internet_options,
        )

    
    st.divider()

    # =====================================================
    # Subscription & Billing
    # =====================================================

    st.subheader("📄 Subscription & Billing")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No",
            ],
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )



    submitted = st.form_submit_button(
    "🔮 Predict Churn",
    use_container_width=True,
    ) 


# -------------------------------------------------------
# Form Submission
# -------------------------------------------------------

if submitted:

    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=10,
            )

        response.raise_for_status()

        result = response.json()

        prediction = result["prediction"]
        probability = float(result["probability"])
        model_version = result["model_version"]

        st.success("Prediction completed successfully!")

        prediction_text = (
            "Likely to Churn"
            if prediction == "Yes"
            else "Not Likely to Churn"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Prediction",
                value=prediction_text,
            )

        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{probability:.2%}",
            )

        with col3:
            st.metric(
                label="Model Version",
                value=model_version,
            )

        st.divider()

        if prediction == "Yes":
            st.error("🔴 This customer is likely to churn.")
        else:
            st.success("🟢 This customer is unlikely to churn.")

        st.subheader("Prediction Confidence")
        st.progress(probability)
        st.caption(f"Estimated churn probability: {probability:.2%}")

        st.divider()

        st.subheader("💡 Recommendations")

        if prediction == "Churn":
            st.warning(
                """
                Based on the prediction, consider the following retention actions:

                - 🎁 Offer a loyalty discount or promotional plan.
                - 📞 Contact the customer proactively.
                - 📄 Recommend switching to a longer-term contract.
                - 🛠️ Offer premium technical support or personalized assistance.
                """
            )
        else:
            st.success(
                """
                This customer appears likely to remain with the service.

                Recommended actions:

                - 😊 Continue delivering excellent customer service.
                - 🎉 Offer loyalty rewards or appreciation benefits.
                - 📦 Promote premium plans or additional services.
                - 📧 Keep the customer engaged through regular communication.
                """
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the prediction API.\n\n"
            "Please ensure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The prediction request timed out."
        )

    except requests.exceptions.HTTPError as e:
        st.error(
            f"API Error ({response.status_code}): {response.text}"
        )

    except Exception as e:
        st.exception(e)