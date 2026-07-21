import sys
from pathlib import Path

import pandas as pd

from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import logger


class DataValidation:
    """
    Responsible for validating the dataset against the expected schema.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation object with its configuration.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validates that the dataset columns exactly match the schema.

        Returns:
            bool: True if validation succeeds, False otherwise.
        """
        try:
            logger.info("Starting data validation...")

            # Read dataset
            data = pd.read_csv(self.config.unzip_data_dir)

            # Dataset columns
            dataset_columns = set(data.columns)

            # Expected columns from schema
            schema_columns = set(self.config.all_schema.keys())

            validation_status = True

            # Find missing columns
            missing_columns = schema_columns - dataset_columns

            # Find unexpected (extra) columns
            extra_columns = dataset_columns - schema_columns

            if missing_columns:
                validation_status = False
                logger.info(f"Missing columns: {sorted(missing_columns)}")

            if extra_columns:
                validation_status = False
                logger.info(f"Unexpected columns: {sorted(extra_columns)}")

            if validation_status:
                logger.info("All columns match the schema.")
            else:
                logger.info("Data validation failed.")

            # Write validation result
            status_file = Path(self.config.STATUS_FILE)

            with open(status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Validation status saved to: {status_file}")

            return validation_status

        except Exception as e:
            logger.exception("Error occurred during data validation.")
            raise CustomException(e, sys)
