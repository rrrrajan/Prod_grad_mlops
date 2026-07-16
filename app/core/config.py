from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from the .env file.
    """

    APP_NAME: str = Field(
        default="Customer Churn Prediction API",
        description="Application name",
    )

    APP_VERSION: str = Field(
        default="1.0.0",
        description="API version",
    )

    API_PREFIX: str = Field(
        default="/api/v1",
        description="API version prefix",
    )

    APP_DESCRIPTION: str = Field(
        default="Production-ready FastAPI service for Customer Churn Prediction.",
    )

    API_CONTACT_NAME: str = Field(
        default="Raghvendra Bhalla",
    )

    API_CONTACT_EMAIL: str = Field(
        default="bhallas.services@gmail.com",
    )

    API_LICENSE_NAME: str = Field(
        default="MIT",
    )

    DEBUG: bool = Field(
        default=False,
        description="Debug mode",
    )

    HOST: str = Field(
        default="0.0.0.0",
    )

    PORT: int = Field(
        default=8000,
    )

    LOG_LEVEL: str = Field(
        default="INFO",
    )

    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Comma-separated list of allowed CORS origins.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        """
        Return the allowed origins as a list.
        """
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()


settings = get_settings()
