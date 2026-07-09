from pathlib import Path

from src.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
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
        Creates and returns the configuration object for the
        Data Ingestion stage.

        Returns
        -------
        DataIngestionConfig
        Configuration containing all paths and parameters
        required for the Data Ingestion component.
        """

        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
        root_dir=Path(config["root_dir"]),

        raw_data_path=Path(config["raw_data_path"]),

        local_data_path=Path(config["local_data_path"]),

        train_data_path=Path(config["train_data_path"]),

        test_data_path=Path(config["test_data_path"]),

        target_column=self.schema["TARGET_COLUMN"],

        random_state=self.params["random_state"],

        test_size=self.params["test_size"],
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
                self.config["data_ingestion"]["local_data_path"]
            ),
            all_schema=self.schema["COLUMNS"],
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
         Creates and returns the configuration object for the
          Data Transformation stage.

         Returns
         -------
         DataTransformationConfig
         Configuration containing all paths and parameters
         required for the Data Transformation component.
        """

        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        data_transformation_config = DataTransformationConfig(root_dir=Path(config["root_dir"]),

           train_data_path=Path(config["train_data_path"]),
           test_data_path=Path(config["test_data_path"]),

           preprocessor_object_path=Path(config["preprocessor_object_path"]),

           transformed_train_path=Path(config["transformed_train_path"]),

           transformed_test_path=Path(config["transformed_test_path"]),

           target_column=self.schema["TARGET_COLUMN"],

           numerical_columns=self.schema["NUMERICAL_COLUMNS"],

           categorical_columns=self.schema["CATEGORICAL_COLUMNS"],

           numeric_conversion_columns=self.schema[
            "NUMERIC_CONVERSION_COLUMNS"],)

        return data_transformation_config