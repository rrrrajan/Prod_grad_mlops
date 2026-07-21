import pytest

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.experiment_tracking.mlflow_tracker import MLflowTracker

pytestmark = pytest.mark.integration


def test_model_evaluation_pipeline():
    """
    Integration test for the complete Model Evaluation stage.
    """

    # Arrange
    config = ConfigurationManager()

    evaluation_config = config.get_model_evaluation_config()
    mlflow_config = config.get_mlflow_config()

    tracker = MLflowTracker(config=mlflow_config)

    evaluator = ModelEvaluation(
        config=evaluation_config,
        tracker=tracker,
    )

    # Act
    results = evaluator.initiate_model_evaluation()

    # Assert
    assert isinstance(results, dict)

    assert "metrics" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results

    metrics = results["metrics"]

    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    # Generated artifacts
    assert evaluation_config.metrics_file_name.exists()
    assert evaluation_config.metadata_file_name.exists()
    assert evaluation_config.classification_report_file_name.exists()
    assert evaluation_config.confusion_matrix_json_file_name.exists()
    assert evaluation_config.confusion_matrix_file_name.exists()
    assert evaluation_config.roc_curve_file_name.exists()
