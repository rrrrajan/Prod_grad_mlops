import tempfile
from pathlib import Path

import mlflow
import pytest

from src.config.configuration import ConfigurationManager
from src.experiment_tracking.mlflow_tracker import MLflowTracker

pytestmark = pytest.mark.integration


def test_mlflow_tracker():
    """
    Integration test for the MLflowTracker.

    Verifies that the tracker can:
      - initialize MLflow
      - start an MLflow run
      - log parameters
      - log metrics
      - set tags
      - log an artifact
      - end the MLflow run
    """

    # Arrange
    config = ConfigurationManager()
    mlflow_config = config.get_mlflow_config()

    tracker = MLflowTracker(config=mlflow_config)

    # Create a temporary artifact
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
    ) as file:
        file.write("MLflow integration test artifact.")
        artifact_path = Path(file.name)

    try:
        # Act
        with tracker.run(run_name="pytest-mlflow-tracker"):

            tracker.log_params(
                {
                    "framework": "pytest",
                    "component": "mlflow_tracker",
                }
            )

            tracker.log_metrics(
                {
                    "accuracy": 0.95,
                    "precision": 0.94,
                    "recall": 0.93,
                }
            )

            tracker.set_tags(
                {
                    "stage": "integration",
                    "author": "pytest",
                }
            )

            tracker.log_artifact(str(artifact_path))

            active_run = mlflow.active_run()

            # Assert run is active
            assert active_run is not None
            assert active_run.info.run_id is not None

        # After exiting the context manager, no run should remain active.
        assert mlflow.active_run() is None

    finally:
        artifact_path.unlink(missing_ok=True)
