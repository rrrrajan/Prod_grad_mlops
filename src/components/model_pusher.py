import shutil
import sys

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    """
    Pushes the approved model and preprocessor to the deployment
    artifacts directory.

    This component copies the trained model and fitted preprocessor
    from their source locations into the Model Pusher directory,
    making them ready for downstream inference or deployment.
    """

    def __init__(self, config: ModelPusherConfig) -> None:
        """
        Initialize the ModelPusher.

        Parameters
        ----------
        config : ModelPusherConfig
            Configuration object containing source and destination paths.
        """
        self.config = config

    def _validate_source_files(self) -> None:
        """
        Validates that the source model and preprocessor exist.

        Raises
        ------
        FileNotFoundError
            If either source file does not exist.
        """
        if not self.config.source_model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.config.source_model_path}"
            )

        if not self.config.source_preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor file not found: "
                f"{self.config.source_preprocessor_path}"
            )

    def _create_destination_directory(self) -> None:
        """
        Creates the destination directory if it does not already exist.
        """
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

    def initiate_model_pusher(self) -> ModelPusherConfig:
        """
        Copies the trained model and preprocessor into the
        deployment directory.

        Returns
        -------
        ModelPusherConfig
            Configuration containing the pushed artifact paths.

        Raises
        ------
        CustomException
            If any error occurs during the model push process.
        """
        try:
            logger.info("Starting Model Pusher component.")

            self._validate_source_files()

            self._create_destination_directory()

            logger.info(
                "Copying trained model from '%s' to '%s'.",
                self.config.source_model_path,
                self.config.pushed_model_path,
            )

            shutil.copy2(
                self.config.source_model_path,
                self.config.pushed_model_path,
            )

            logger.info(
                "Copying preprocessor from '%s' to '%s'.",
                self.config.source_preprocessor_path,
                self.config.pushed_preprocessor_path,
            )

            shutil.copy2(
                self.config.source_preprocessor_path,
                self.config.pushed_preprocessor_path,
            )

            logger.info("Model Pusher completed successfully.")

            return self.config

        except Exception as e:
            logger.exception("Error occurred during Model Pusher.")
            raise CustomException(e, sys)
