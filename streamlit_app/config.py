"""
Configuration settings for the Streamlit application.
"""
API_BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"

PREDICT_ENDPOINT = f"{API_BASE_URL}{API_PREFIX}/predict"



API_URL = "http://localhost:8000/predict"
REQUEST_TIMEOUT = 20

APP_TITLE = "Customer Churn Prediction"

APP_ICON = "📈"