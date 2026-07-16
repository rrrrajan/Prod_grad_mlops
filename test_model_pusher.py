import sys

from src.logger import logger
from src.exception import CustomException

from src.pipeline.stage_06_model_pusher import (
    ModelPusherTrainingPipeline,
)

if __name__ == "__main__":
    try:
        logger.info("*" * 80)
        logger.info("Stage 06: Model Pusher started")
        logger.info("*" * 80)

        obj = ModelPusherTrainingPipeline()
        obj.main()

        logger.info("*" * 80)
        logger.info("Stage 06: Model Pusher completed successfully")
        logger.info("*" * 80)

    except Exception as e:
        logger.exception("Error occurred while testing Model Pusher.")
        raise CustomException(e, sys)
