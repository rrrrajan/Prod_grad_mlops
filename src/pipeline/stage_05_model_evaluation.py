from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.exception import CustomException
from src.logger import logger
import sys
from src.experiment_tracking.mlflow_tracker import MLflowTracker
from src.experiment_tracking.experiment_tracker import ExperimentTracker


class ModelEvaluationPipeline:
    """
    Pipeline class for the Model Evaluation stage.

    This pipeline is responsible for:
        1. Loading the Model Evaluation configuration.
        2. Initializing the ModelEvaluation component.
        3. Executing the Model Evaluation stage.
    """

    def __init__(self):
        """Initialize the Model Evaluation pipeline."""
        pass

    def run(self) -> None:
        """
        Execute the complete Model Evaluation pipeline.
        """

        try:
            logger.info("=" * 60)

            logger.info("Starting Model Evaluation Pipeline.")

            config_manager = ConfigurationManager()

            mlflow_config = config_manager.get_mlflow_config()

            tracker: ExperimentTracker = MLflowTracker(mlflow_config)

            model_evaluation_config = config_manager.get_model_evaluation_config()

            model_evaluation = ModelEvaluation(
                config=model_evaluation_config,
                tracker=tracker,
            )

            metrics = model_evaluation.initiate_model_evaluation()

            logger.info("Model Evaluation Pipeline completed successfully.")

            logger.info("=" * 60)

            logger.info(
                "Evaluation Summary : %s",
                metrics,
            )

        except Exception as e:
            logger.exception("Error occurred in Model Evaluation Pipeline.")
            raise CustomException(e, sys)
