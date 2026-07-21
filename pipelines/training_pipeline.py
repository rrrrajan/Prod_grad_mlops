"""
Training Pipeline

This module orchestrates the complete model training workflow.

Current stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

"""

from src.logger import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import \
    DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import \
    DataTransformationTrainingPipeline
from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


class TrainingPipeline:
    """
    Orchestrates the end-to-end model training workflow.
    """

    def __init__(self):
        logger.info("Initializing Training Pipeline...")

    def run(self) -> None:

        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 80)

        DataIngestionTrainingPipeline().run()

        DataValidationTrainingPipeline().run()

        DataTransformationTrainingPipeline().run()

        ModelTrainerTrainingPipeline().run()

        ModelEvaluationPipeline().run()

        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 80)


if __name__ == "__main__":
    TrainingPipeline().run()
