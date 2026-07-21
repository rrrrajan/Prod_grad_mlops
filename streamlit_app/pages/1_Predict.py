import requests
import streamlit as st

from config import HEALTH_ENDPOINT, PREDICT_ENDPOINT

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Customer Churn MLOps")

try:
    health_response = requests.get(
        HEALTH_ENDPOINT,
        timeout=3,
    )

    if health_response.status_code == 200:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Unavailable")

except requests.exceptions.RequestException:
    st.sidebar.error("🔴 API Offline")

st.sidebar.divider()

st.sidebar.caption("FastAPI • Streamlit • Docker • MLflow")

# -------------------------------------------------------
# Main Page
# -------------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.markdown("""
Predict whether a telecom customer is likely to churn based on
their demographic information and subscribed services.
""")

st.divider()

# -------------------------------------------------------
# Prediction Form
# -------------------------------------------------------

with st.form("prediction_form", clear_on_submit=False):

    # =====================================================
    # Customer Information
    # =====================================================

    st.subheader("👤 Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            label="Gender",
            options=["Male", "Female"],
            help="Customer gender",
        )

        senior_citizen = st.selectbox(
            label="Senior Citizen",
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
            help="Whether the customer is a senior citizen",
        )

        partner = st.selectbox(
            label="Partner",
            options=["Yes", "No"],
        )

        dependents = st.selectbox(
            label="Dependents",
            options=["Yes", "No"],
        )

    with col2:

        tenure = st.number_input(
            label="Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            help="Number of months the customer has stayed with the company",
        )

        monthly_charges = st.number_input(
            label="Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            format="%.2f",
        )

        total_charges = st.number_input(
            label="Total Charges ($)",
            min_value=0.0,
            value=850.0,
            step=10.0,
            format="%.2f",
        )

    st.divider()

    # =====================================================
    # Phone Services
    # =====================================================

    st.subheader("☎️ Phone Services")

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            label="Phone Service",
            options=["Yes", "No"],
            help="Whether the customer has a phone service",
        )

    with col2:

        multiple_lines = st.selectbox(
            label="Multiple Lines",
            options=(["No phone service"] if phone_service == "No" else ["No", "Yes"]),
            help="Whether the customer has multiple phone lines",
        )

    st.divider()

    # =====================================================
    # Internet Services
    # =====================================================

    st.subheader("🌐 Internet Services")

    col1, col2 = st.columns(2)

    with col1:

        internet_service = st.selectbox(
            label="Internet Service",
            options=[
                "DSL",
                "Fiber optic",
                "No",
            ],
            help="Customer's internet connection type",
        )

        internet_options = (
            ["No internet service"] if internet_service == "No" else ["No", "Yes"]
        )

        online_security = st.selectbox(
            label="Online Security",
            options=internet_options,
        )

        online_backup = st.selectbox(
            label="Online Backup",
            options=internet_options,
        )

        device_protection = st.selectbox(
            label="Device Protection",
            options=internet_options,
        )

    with col2:

        tech_support = st.selectbox(
            label="Tech Support",
            options=internet_options,
        )

        streaming_tv = st.selectbox(
            label="Streaming TV",
            options=internet_options,
        )

        streaming_movies = st.selectbox(
            label="Streaming Movies",
            options=internet_options,
        )

    st.divider()

    # =====================================================
    # Subscription & Billing
    # =====================================================

    st.subheader("📄 Subscription & Billing")

    col1, col2 = st.columns(2)

    with col1:

        contract = st.selectbox(
            label="Contract",
            options=[
                "Month-to-month",
                "One year",
                "Two year",
            ],
            help="Customer contract type",
        )

        paperless_billing = st.selectbox(
            label="Paperless Billing",
            options=[
                "Yes",
                "No",
            ],
        )

    with col2:

        payment_method = st.selectbox(
            label="Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.divider()

    # =====================================================
    # Form Buttons
    # =====================================================

    btn1, btn2 = st.columns(2)

    with btn1:

        submitted = st.form_submit_button(
            label="🔮 Predict Churn",
            use_container_width=True,
        )

    with btn2:

        reset = st.form_submit_button(
            label="🔄 Reset",
            use_container_width=True,
        )

# -------------------------------------------------------
# Reset Form
# -------------------------------------------------------

if reset:
    st.rerun()

# -------------------------------------------------------
# Basic Input Validation
# -------------------------------------------------------

if submitted:

    if monthly_charges < 0:
        st.error("Monthly Charges cannot be negative.")
        st.stop()

    if total_charges < 0:
        st.error("Total Charges cannot be negative.")
        st.stop()

    if tenure > 0 and total_charges == 0:
        st.warning("Total Charges seem unusually low for the given tenure.")

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

        # -------------------------------------------------------
        # Call Prediction API
        # -------------------------------------------------------

        with st.spinner("Running customer churn prediction..."):

            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=10,
            )

        response.raise_for_status()

        result = response.json()

        # Store latest prediction
        st.session_state["last_prediction"] = result

        # -------------------------------------------------------
        # Parse Response
        # -------------------------------------------------------

        prediction = result.get("prediction", "").strip().lower()

        probability = float(result.get("probability", 0.0))

        probability = max(
            0.0,
            min(probability, 1.0),
        )

        model_version = result.get(
            "model_version",
            "N/A",
        )

        is_churn = prediction == "churn"

        prediction_text = "Likely to Churn" if is_churn else "Not Likely to Churn"

        # -------------------------------------------------------
        # Success Message
        # -------------------------------------------------------

        st.success("Prediction completed successfully!")

        # -------------------------------------------------------
        # Metrics
        # -------------------------------------------------------

        col1, col2, col3 = st.columns([2, 2, 1])

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

        # -------------------------------------------------------
        # Prediction Result
        # -------------------------------------------------------

        if is_churn:

            st.error("🔴 This customer is likely to churn.")

        else:

            st.success("🟢 This customer is unlikely to churn.")

        # -------------------------------------------------------
        # Confidence
        # -------------------------------------------------------

        st.subheader("Prediction Confidence")

        st.progress(probability)

        if probability >= 0.70:

            st.error(f"Confidence: {probability:.2%}")

        elif probability >= 0.40:

            st.warning(f"Confidence: {probability:.2%}")

        else:

            st.success(f"Confidence: {probability:.2%}")

        st.divider()

        # -------------------------------------------------------
        # Prediction Details
        # -------------------------------------------------------

        with st.expander("Prediction Details"):

            st.json(result)

        st.divider()

        # -------------------------------------------------------
        # Recommendations
        # -------------------------------------------------------

        st.subheader("💡 Recommendations")

        if is_churn:

            st.warning("""
Based on the prediction, consider the following retention actions:

• 🎁 Offer a loyalty discount.

• 📞 Contact the customer proactively.

• 📄 Recommend a longer-term contract.

• 🛠️ Offer premium technical support.

• 💳 Provide attractive payment options.

• ⭐ Enroll the customer in a loyalty program.
""")

        else:

            st.success("""
This customer appears likely to remain with the service.

Recommended actions:

• 😊 Continue delivering excellent customer service.

• 🎉 Offer loyalty rewards.

• 📦 Recommend premium plans.

• 📧 Keep the customer engaged through regular communication.

• ⭐ Monitor satisfaction regularly.
""")

    # -------------------------------------------------------
    # Connection Error
    # -------------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(f"""
Unable to connect to the Prediction API.

Expected endpoint:
{PREDICT_ENDPOINT}

Please verify that:

• FastAPI server is running
• Docker container is running
• API URL in config.py is correct
""")

    # -------------------------------------------------------
    # Timeout
    # -------------------------------------------------------

    except requests.exceptions.Timeout:

        st.error("""
The prediction request timed out.

Please try again in a few seconds.
""")

    # -------------------------------------------------------
    # HTTP Error
    # -------------------------------------------------------

    except requests.exceptions.HTTPError:

        st.error(f"API Error ({response.status_code})")

        try:
            st.json(response.json())
        except Exception:
            st.code(response.text)

    # -------------------------------------------------------
    # Invalid JSON
    # -------------------------------------------------------

    except ValueError:

        st.error("The API returned an invalid response.")

    # -------------------------------------------------------
    # Unexpected Error
    # -------------------------------------------------------

    except Exception as e:

        st.exception(e)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction System | " "FastAPI • Streamlit • Docker • MLflow"
)
