import sys

from src.config.configuration import ConfigurationManager
from src.components.deployment import Deployment
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

            deployment = Deployment(deployment_config)

            # Step 1: Connect to MLflow
            deployment.connect_to_mlflow()

            # Step 2: Get latest registered model
            latest = deployment.get_latest_model_version()

            print(f"Version : {latest.version}")
            print(f"Run ID  : {latest.run_id}")
            print(f"Stage   : {latest.current_stage}")

            logger.info(f">>>>>> Stage completed: {STAGE_NAME} <<<<<<\n")

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)