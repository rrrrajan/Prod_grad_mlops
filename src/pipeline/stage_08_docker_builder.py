from src.config.configuration import ConfigurationManager
from src.components.docker_builder import DockerBuilder

from src.logger import logger
from src.exception import CustomException

import sys


STAGE_NAME = "Docker Builder Stage"


class DockerBuilderPipeline:
    """
    Pipeline for the Docker Builder stage.
    """

    def __init__(self):
        pass

    def main(self) -> None:
        """
        Executes the Docker Builder stage.
        """

        try:
            config = ConfigurationManager()

            docker_builder_config = (
                config.get_docker_builder_config()
            )

            docker_builder = DockerBuilder(
                docker_builder_config
            )

            docker_builder.build_image()

            logger.info(
                "Docker Builder stage completed successfully."
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:

        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")

        obj = DockerBuilderPipeline()

        obj.main()

        logger.info(
            f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x"
        )

    except Exception as e:

        logger.exception(e)

        raise e