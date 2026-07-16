from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from app.middleware.exception_handler import register_exception_handlers

from app.middleware.logging_middleware import LoggingMiddleware

from app.core.config import settings

from app.api.v1.routes import router

from fastapi import FastAPI, Request

from src.logger import logger
from src.pipeline.prediction_pipeline import (
    PredictionPipeline,
)


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
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    logger.info("=" * 60)

    try:
        app.state.prediction_pipeline = PredictionPipeline()

        logger.info("Prediction pipeline loaded successfully.")

    except Exception:
        logger.exception("Failed to initialize PredictionPipeline.")
        raise

    yield

    logger.info("=" * 60)
    logger.info("Shutting down %s...", settings.APP_NAME)
    logger.info("=" * 60)


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    contact={
        "name": settings.API_CONTACT_NAME,
        "email": settings.API_CONTACT_EMAIL,
    },
    license_info={
        "name": settings.API_LICENSE_NAME,
    },
    lifespan=lifespan,
)

register_exception_handlers(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/")
def home() -> dict[str, str]:
    """
    Root endpoint.
    """
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
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
