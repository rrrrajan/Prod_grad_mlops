import pytest

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager

pytestmark = pytest.mark.integration


def test_model_evaluation_pipeline():
    """
    Integration test for the complete Model Evaluation stage.
    """

    # Arrange
    config = ConfigurationManager()
    evaluation_config = config.get_model_evaluation_config()

    evaluator = ModelEvaluation(config=evaluation_config)

    # Act
    results = evaluator.initiate_model_evaluation()

    # Assert returned object
    assert isinstance(results, dict)

    assert "metrics" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results

    metrics = results["metrics"]

    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    # Check evaluation artifacts
    assert evaluation_config.metrics_file_name.exists()
    assert evaluation_config.metadata_file_name.exists()
    assert evaluation_config.classification_report_file_name.exists()
    assert evaluation_config.confusion_matrix_json_file_name.exists()
    assert evaluation_config.confusion_matrix_file_name.exists()
    assert evaluation_config.roc_curve_file_name.exists()