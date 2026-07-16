from src.logger import logger
from src.exception import CustomException

from src.config.configuration import ConfigurationManager
from src.components.model_pusher import ModelPusher
import sys


class ModelPusherTrainingPipeline:
    """
    Training pipeline for the Model Pusher stage.

    This pipeline is responsible for copying the approved model and
    preprocessor from the training artifacts directory to the deployment
    artifacts directory.
    """

    def __init__(self) -> None:
        """Initialize the Model Pusher training pipeline."""
        pass

    def main(self) -> None:
        """
        Execute the Model Pusher pipeline.

        Raises
        ------
        CustomException
            If any error occurs during execution.
        """
        try:
            logger.info("========== Stage 06: Model Pusher started ==========")

            config_manager = ConfigurationManager()

            model_pusher_config = config_manager.get_model_pusher_config()

            model_pusher = ModelPusher(config=model_pusher_config)

            model_pusher.initiate_model_pusher()

            logger.info("========== Stage 06: Model Pusher completed ==========")

        except Exception as e:
            logger.exception("Error occurred during Stage 06: Model Pusher.")
            raise CustomException(e, sys)
