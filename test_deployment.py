from src.components.deployment import Deployment
from src.config.configuration import ConfigurationManager
from src.logger import logger

STAGE_NAME = "Deployment Stage"


class DeploymentPipeline:
    """
    Pipeline for the Deployment stage.
    """

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        deployment_config = config.get_deployment_config()
        mlflow_config = config.get_mlflow_config()

        deployment = Deployment(
            deployment_config=deployment_config,
            mlflow_config=mlflow_config,
        )

        deployment.deploy()


if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)

    try:
        DeploymentPipeline().main()

        logger.info(
            ">>>>>> %s completed <<<<<<",
            STAGE_NAME,
        )

    except Exception:
        logger.exception(
            ">>>>>> %s failed <<<<<<",
            STAGE_NAME,
        )
        raise
