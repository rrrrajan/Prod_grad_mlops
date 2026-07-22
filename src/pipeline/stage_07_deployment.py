import sys

from src.components.deployment import Deployment
from src.config.configuration import ConfigurationManager
from src.exception import CustomException
from src.logger import logger

STAGE_NAME = "Deployment Stage"


class DeploymentPipeline:
    """
    Pipeline for the Deployment stage.
    """

    def __init__(self):
        pass

    def run(self) -> None:
        """
        Executes the Deployment stage.
        """
        try:
            logger.info(f">>>>>> Stage started: {STAGE_NAME} <<<<<<")

            config = ConfigurationManager()

            deployment_config = config.get_deployment_config()
            mlflow_config = config.get_mlflow_config()

            deployment = Deployment(
                deployment_config=deployment_config,
                mlflow_config=mlflow_config,
            )

            # Connect to MLflow and download the latest registered model
            deployment.prepare_model()

            logger.info(f">>>>>> Stage completed: {STAGE_NAME} <<<<<<\n")

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DeploymentPipeline()
    obj.run()
