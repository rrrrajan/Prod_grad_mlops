import sys

from src.logger import logger
from src.exception import CustomException

from src.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

from src.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)

from src.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)

from src.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)


# ===========================
# Stage 01 : Data Ingestion
# ===========================

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# ===========================
# Stage 02 : Data Validation
# ===========================

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = DataValidationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# ===============================
# Stage 03 : Data Transformation
# ===============================

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = DataTransformationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# ==========================
# Stage 04 : Model Trainer
# ==========================

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    obj = ModelTrainerTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)