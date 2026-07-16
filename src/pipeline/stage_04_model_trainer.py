from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.logger import logger

from src.exception import CustomException
import sys

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    """
    Pipeline for the Model Trainer stage.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Executes the Model Trainer stage.
        """
        config = ConfigurationManager()

        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(config=model_trainer_config)

        model_trainer.initiate_model_trainer()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

        obj = ModelTrainerTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
