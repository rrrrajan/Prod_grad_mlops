from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any
from collections.abc import Iterable


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking backends.

    Concrete implementations may include:
    - MLflow
    - Weights & Biases
    - Neptune
    - TensorBoard
    - DummyTracker (no-op)
    """

    @abstractmethod
    def run(
        self,
        run_name: str | None = None,
    ) -> AbstractContextManager[None]:
        """
        Return a context manager that automatically starts and ends
        an experiment run.

        Usage:
            with tracker.run("RandomForest"):
                ...
        """
        ...

    @abstractmethod
    def start_run(self, run_name: str | None = None) -> None:
        """Start a new experiment run."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """End the active experiment run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log model or training parameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics."""
        ...

    @abstractmethod
    def log_artifact(self, artifact_path: str) -> None:
        """Log a single artifact."""
        ...

    @abstractmethod
    def log_artifacts(self, artifact_paths: Iterable[str]) -> None:
        """Log multiple artifacts."""
        ...

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
    ) -> None:
        """Log the trained model."""
        ...

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """Set metadata tags for the experiment."""
        ...
