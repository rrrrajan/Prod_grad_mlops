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

from src.pipeline.stage_05_model_evaluation import (
    ModelEvaluationPipeline,
)

from src.pipeline.stage_06_model_pusher import (
    ModelPusherTrainingPipeline,
)

from src.pipeline.stage_07_deployment import (
    DeploymentPipeline,
)

# ===========================
# Stage 01 : Data Ingestion
# ===========================

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = DataIngestionTrainingPipeline()
    pipeline.run()

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

    pipeline = DataValidationTrainingPipeline()
    pipeline.run()

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

    pipeline = DataTransformationTrainingPipeline()
    pipeline.run()

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

    pipeline = ModelTrainerTrainingPipeline()
    pipeline.run()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# =============================
# Stage 05 : Model Evaluation
# =============================

STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = ModelEvaluationPipeline()
    pipeline.run()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# =========================
# Stage 06 : Model Pusher
# =========================

STAGE_NAME = "Model Pusher Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = ModelPusherTrainingPipeline()
    pipeline.run()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)


# =========================
# Stage 07 : Deployment
# =========================

STAGE_NAME = "Deployment Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = DeploymentPipeline()
    pipeline.run()

    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)