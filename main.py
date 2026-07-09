from src.logger import logger
from src.exception import CustomException
import sys

from src.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

from src.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = DataIngestionTrainingPipeline()

    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = DataValidationTrainingPipeline()

    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)