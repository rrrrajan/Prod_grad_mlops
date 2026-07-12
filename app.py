from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from src.logger import logger
from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictionPipeline,
)
from src.schema.request import CustomerRequest
from src.schema.response import PredictionResponse


def get_prediction_pipeline(request: Request) -> PredictionPipeline:
    """
    Return the PredictionPipeline instance stored in the FastAPI
    application state.
    """
    return request.app.state.prediction_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the prediction pipeline once during application startup
    and reuse it for all incoming prediction requests.
    """

    logger.info("=" * 60)
    logger.info("Starting Customer Churn Prediction API...")
    logger.info("=" * 60)

    try:
        app.state.prediction_pipeline = PredictionPipeline()

        logger.info("Prediction pipeline loaded successfully.")

    except Exception:
        logger.exception("Failed to initialize PredictionPipeline.")
        raise

    yield

    logger.info("=" * 60)
    logger.info("Shutting down Customer Churn Prediction API...")
    logger.info("=" * 60)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready FastAPI service for Customer Churn Prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict[str, str]:
    """
    Root endpoint.
    """
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    customer: CustomerRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
) -> PredictionResponse:
    """
    Predict customer churn.
    """

    try:
        logger.info("Received prediction request.")

        custom_data = CustomData(
            gender=customer.gender,
            SeniorCitizen=customer.SeniorCitizen,
            Partner=customer.Partner,
            Dependents=customer.Dependents,
            tenure=customer.tenure,
            PhoneService=customer.PhoneService,
            MultipleLines=customer.MultipleLines,
            InternetService=customer.InternetService,
            OnlineSecurity=customer.OnlineSecurity,
            OnlineBackup=customer.OnlineBackup,
            DeviceProtection=customer.DeviceProtection,
            TechSupport=customer.TechSupport,
            StreamingTV=customer.StreamingTV,
            StreamingMovies=customer.StreamingMovies,
            Contract=customer.Contract,
            PaperlessBilling=customer.PaperlessBilling,
            PaymentMethod=customer.PaymentMethod,
            MonthlyCharges=customer.MonthlyCharges,
            TotalCharges=customer.TotalCharges,
        )

        features = custom_data.get_data_as_dataframe()

        logger.info("Input successfully converted to DataFrame.")

        result = pipeline.predict(features)

        logger.info(
            "Prediction completed successfully. "
            "Prediction=%s, Label=%s, Probability=%s",
            result["prediction"],
            result["label"],
            result["probability"],
        )

        return PredictionResponse(
            prediction=result["label"],
            probability=result["probability"],
            model_version="1.0.0",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )