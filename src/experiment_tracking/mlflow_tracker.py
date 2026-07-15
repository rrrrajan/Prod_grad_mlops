from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from src.entity.config_entity import MLflowConfig
from src.experiment_tracking.experiment_tracker import ExperimentTracker
from src.logger import logger

class MLflowTracker(ExperimentTracker):
    """
    MLflow implementation of the ExperimentTracker interface.
    """

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config

        if self.config.enabled:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)




    @contextmanager
    def run(self, run_name: str | None = None) -> Generator[None, None, None]:
        """
        Context manager for an MLflow run.
        """

        self.start_run(run_name)

        try:
            yield

        finally:
            logger.debug("Closing MLflow run.")

            self.end_run()


    def start_run(self, run_name: str | None = None) -> None:
        """
        Start an MLflow run.
        """
       
        if not self.config.enabled:
           return 

        logger.info(
          "Starting MLflow run: %s",
          run_name)

        mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """
        End the active MLflow run.
        """
        if not self.config.enabled:
           return 
            
        logger.info("Ending MLflow run.")
        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.config.enabled:
           return
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Log evaluation metrics to MLflow.
        """
        if not self.config.enabled:
            return
        mlflow.log_metrics(metrics)

    def log_artifact(self, artifact_path: str) -> None:
        if not self.config.enabled:
           return
           
        mlflow.log_artifact(artifact_path)

    def log_artifacts(self, artifact_paths: Iterable[str]) -> None:
        """
        Log multiple artifacts to MLflow.
        """

        if not self.config.enabled:
            return

        for artifact_path in artifact_paths:
            if Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)
            else:
                logger.warning(
                    "Artifact not found and will not be logged: %s",
                    artifact_path,
                )

    
    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """
        Log the trained model to MLflow.
        """

        if not self.config.enabled:
            return

        logger.info("Logging trained model to MLflow.")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
        )


    def set_tags(self, tags: dict[str, str]) -> None:
        if not self.config.enabled:
            return
            
        mlflow.set_tags(tags)