from abc import ABC, abstractmethod
from typing import Any


class ModelRegistry(ABC):
    """
    Abstract interface for model registries.
    """

    @abstractmethod
    def register_model(self, model_uri: str, model_name: str) -> Any:
        """Register a model."""
        ...

    @abstractmethod
    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        """Assign an alias to a model version."""
        ...

    @abstractmethod
    def get_model_version(self, model_name: str, alias: str) -> Any:
        """Retrieve a model version by alias."""
        ...

    @abstractmethod
    def download_model(self, model_uri: str, dst_path: str | None = None) -> str:
        """Download a registered model."""
        ...
