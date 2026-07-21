import sys

from src.components.data_transformation import DataTransformation
from src.config.configuration import ConfigurationManager
from src.exception import CustomException
from src.logger import logger

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    """
    Pipeline for the Data Transformation stage.
    """

    def __init__(self):
        pass

    def run(self) -> None:
        """
        Executes the Data Transformation stage.
        """
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)

        data_transformation.initiate_data_transformation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")

        obj = DataTransformationTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
