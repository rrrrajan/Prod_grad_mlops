import sys

from src.exception import CustomException
from src.logger import logger

from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_06_model_pusher import ModelPusherTrainingPipeline


def run_stage(stage_name: str, pipeline_class) -> None:
    """
    Execute a pipeline stage with consistent logging and error handling.
    """
    try:
        logger.info("=" * 60)
        logger.info(">>>>>> %s started <<<<<<", stage_name)
        logger.info("=" * 60)

        pipeline = pipeline_class()
        pipeline.run()

        logger.info(">>>>>> %s completed <<<<<<", stage_name)
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.exception("%s failed.", stage_name)
        raise CustomException(e, sys)


def main() -> None:
    """
    Run the complete model training pipeline.

    Stages:
        01 - Data Ingestion
        02 - Data Validation
        03 - Data Transformation
        04 - Model Trainer
        05 - Model Evaluation
        06 - Model Pusher (MLflow Model Registry)

    Deployment is intentionally excluded.
    """

    stages = [
        ("Data Ingestion Stage", DataIngestionTrainingPipeline),
        ("Data Validation Stage", DataValidationTrainingPipeline),
        ("Data Transformation Stage", DataTransformationTrainingPipeline),
        ("Model Trainer Stage", ModelTrainerTrainingPipeline),
        ("Model Evaluation Stage", ModelEvaluationPipeline),
        ("Model Pusher Stage", ModelPusherTrainingPipeline),
    ]

    logger.info("Starting Training Pipeline...")

    for stage_name, pipeline_class in stages:
        run_stage(stage_name, pipeline_class)

    logger.info("=" * 60)
    logger.info("Training Pipeline completed successfully.")
    logger.info("=" * 60)
    
if __name__ == "__main__":
    main()