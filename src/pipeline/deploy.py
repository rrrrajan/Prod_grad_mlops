import sys

from src.exception import CustomException
from src.logger import logger
from src.pipeline.stage_07_deployment import DeploymentPipeline


def run_stage(stage_name: str, pipeline_class) -> None:
    """
    Execute a pipeline stage with consistent logging and error handling.
    """
    try:
        logger.info("=" * 60)
        logger.info(">>>>>> %s started <<<<<<", stage_name)
        logger.info("=" * 60)

        pipeline = pipeline_class()
        pipeline.run()

        logger.info(">>>>>> %s completed <<<<<<", stage_name)
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.exception("%s failed.", stage_name)
        raise CustomException(e, sys)


def main() -> None:
    """
    Deploy the latest registered model.

    This pipeline performs the deployment stage only:
        - Connect to MLflow
        - Download the latest registered model
        - Build Docker image
        - Start/Restart the API container
        - Perform health check
    """

    logger.info("Starting Deployment Pipeline...")

    run_stage(
        "Deployment Stage",
        DeploymentPipeline,
    )

    logger.info("=" * 60)
    logger.info("Deployment Pipeline completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
