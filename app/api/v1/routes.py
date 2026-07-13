from fastapi import APIRouter, Depends, Request

from fastapi import HTTPException

from src.logger import logger
from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictionPipeline,
)
from src.schema.request import CustomerRequest
from src.schema.response import PredictionResponse
from app.core.config import settings

def get_prediction_pipeline(request: Request) -> PredictionPipeline:
    """
    Return the PredictionPipeline instance stored in the
    FastAPI application state.
    """
    return request.app.state.prediction_pipeline


router = APIRouter()


@router.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
)

@router.get(
    "/ready",
    tags=["Health"],
    summary="Readiness Check",
)
def readiness(
    request: Request,
) -> dict[str, str]:
    """
    Readiness endpoint.

    Checks whether the prediction pipeline
    has been initialized successfully.
    """

    if not hasattr(request.app.state, "prediction_pipeline"):
        raise HTTPException(
            status_code=503,
            detail="Prediction pipeline is not ready.",
        )

    return {
        "status": "ready",
    }



def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
    }

@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Customer Churn",
)


def predict(customer: CustomerRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline)) -> PredictionResponse:
    """
    Predict customer churn.
    """

    
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
        model_version=settings.APP_VERSION,
    )
      

