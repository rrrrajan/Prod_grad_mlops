import sys

from src.exception import CustomException
from src.logger import logger
from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Model Evaluation Stage"


try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = ModelEvaluationPipeline()

    pipeline.run()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)
