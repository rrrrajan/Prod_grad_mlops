from pathlib import Path
from src.exception import CustomException
import sys
from src.logger import logger


from src.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    PredictionConfig,
    MLflowConfig,
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
            unzip_data_dir=Path(self.config["data_ingestion"]["local_data_path"]),
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

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            preprocessor_object_path=Path(config["preprocessor_object_path"]),
            transformed_train_path=Path(config["transformed_train_path"]),
            transformed_test_path=Path(config["transformed_test_path"]),
            target_column=self.schema["TARGET_COLUMN"],
            numerical_columns=self.schema["NUMERICAL_COLUMNS"],
            categorical_columns=self.schema["CATEGORICAL_COLUMNS"],
            numeric_conversion_columns=self.schema["NUMERIC_CONVERSION_COLUMNS"],
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Creates and returns the configuration object for the
        Model Trainer stage.

        Returns
        -------
        ModelTrainerConfig
        Configuration containing all paths required
        by the Model Trainer component.
        """

        config = self.config["model_trainer"]
        params = self.params
        transformation_config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config["root_dir"]),
            transformed_train_path=Path(
                transformation_config["transformed_train_path"]
            ),
            transformed_test_path=Path(transformation_config["transformed_test_path"]),
            trained_model_path=Path(config["trained_model_path"]),
            metrics_file_name=Path(config["metrics_file_name"]),
            model_report_file_name=Path(config["model_report_file_name"]),
            evaluation_metric=params["model_trainer"]["evaluation_metric"],
            model_params=params["model_trainer"]["models"],
        )  # <-- Pass it here

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Creates and returns the configuration object for the
        Model Evaluation stage.

        Returns
        -------
        ModelEvaluationConfig
            Configuration containing all paths and parameters
            required for the Model Evaluation component.
        """

        config = self.config["model_evaluation"]
        params = self.params
        schema = self.schema

        create_directories([config["root_dir"]])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config["root_dir"]),
            trained_model_path=Path(config["trained_model_path"]),
            test_array_path=Path(config["test_data_path"]),
            metrics_file_name=Path(config["metrics_file_name"]),
            metadata_file_name=Path(config["metadata_file_name"]),
            classification_report_file_name=Path(
                config["classification_report_file_name"]
            ),
            confusion_matrix_json_file_name=Path(
                config["confusion_matrix_json_file_name"]
            ),
            roc_curve_file_name=Path(config["roc_curve_file_name"]),
            confusion_matrix_file_name=Path(config["confusion_matrix_file_name"]),
            target_column=schema["TARGET_COLUMN"],
            evaluation_metric=params["model_trainer"]["evaluation_metric"],
            random_state=params["random_state"],
            experiment_name=params["model_evaluation"]["experiment_name"],
            run_name=params["model_evaluation"]["run_name"],
            log_model=params["model_evaluation"]["log_model"],
            register_model=params["model_evaluation"]["register_model"],
            registered_model_name=params["model_evaluation"]["registered_model_name"],
        )

        return model_evaluation_config

    def get_model_pusher_config(self) -> ModelPusherConfig:
        """
        Creates and returns the configuration object for the
        Model Pusher stage.

        Returns
        -------
        ModelPusherConfig
            Configuration containing all paths required
            for the Model Pusher component.
        """

        config = self.config["model_pusher"]

        create_directories([config["root_dir"]])

        model_pusher_config = ModelPusherConfig(
            root_dir=Path(config["root_dir"]),
            source_model_path=Path(config["source_model_path"]),
            source_preprocessor_path=Path(config["source_preprocessor_path"]),
            pushed_model_path=Path(config["pushed_model_path"]),
            pushed_preprocessor_path=Path(config["pushed_preprocessor_path"]),
        )

        return model_pusher_config

    def get_prediction_config(self) -> PredictionConfig:
        """
        Creates and returns the PredictionConfig required for inference.
        """

        try:
            config = self.config["prediction"]

            prediction_config = PredictionConfig(
                model_path=Path(config["model_path"]),
                preprocessor_path=Path(config["preprocessor_path"]),
            )

            logger.info("Prediction configuration loaded successfully.")

            return prediction_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_mlflow_config(self) -> MLflowConfig:
        """
        Returns MLflow configuration.
        """

        config = self.config["mlflow"]

        return MLflowConfig(
            enabled=config["enabled"],
            tracking_uri=config["tracking_uri"],
            experiment_name=config["experiment_name"],
        )
