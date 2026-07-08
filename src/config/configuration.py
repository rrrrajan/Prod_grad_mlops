from pathlib import Path

from src.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)

from src.utils.common import read_yaml, create_directories


class ConfigurationManager:
    """
    Reads and manages all project configuration files.
    """

    def __init__(self):
        """
        Load all YAML configuration files once when the object is created.
        """

        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        # Create the root artifacts directory
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the configuration required for the Data Ingestion component.
        """

        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            raw_data_path=Path(config["raw_data_path"]),
            local_data_file=Path(config["local_data_file"]),
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Returns the configuration required for the Data Validation component.
        """

        config = self.config["data_validation"]

        create_directories([config["root_dir"]])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config["root_dir"]),
            STATUS_FILE=Path(config["STATUS_FILE"]),
            unzip_data_dir=Path(
                self.config["data_ingestion"]["local_data_file"]
            ),
            all_schema=self.schema["COLUMNS"],
        )

        return data_validation_config