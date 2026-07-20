import streamlit as st

from config import APP_TITLE, APP_ICON

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)

st.title("📈 Customer Churn Prediction")

st.markdown("---")

st.write(
    """
    Welcome to the Customer Churn Prediction application.

    This application is built using:

    - FastAPI
    - Streamlit
    - MLflow
    - Docker
    - Production MLOps Pipeline

    Use the navigation menu on the left to make predictions.
    """
)