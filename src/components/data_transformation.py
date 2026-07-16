import pandas as pd
import numpy as np
from pathlib import Path

from src.exception import CustomException
import sys

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.logger import logger
from src.utils.common import save_object


class DataTransformation:
    """
    Responsible for transforming validated datasets
    into machine-learning-ready datasets.
    """

    def __init__(self, config: DataTransformationConfig) -> None:
        """
        Initialize the Data Transformation component.

        Parameters
        ----------
        config : DataTransformationConfig
            Configuration required for data transformation.
        """

        self.config = config

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the validated train and test datasets.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Training and testing dataframes.
        """

        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        logger.info("Training dataset loaded successfully.")
        logger.info("Testing dataset loaded successfully.")

        return train_df, test_df

    def clean_numeric_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert configured columns from object to numeric.

        Invalid values are converted to NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.
        """

        df = df.copy()

        for column in self.config.numeric_conversion_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df

    def split_features_target(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataframe into features and target.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature matrix and target vector.
        """

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        return X, y

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline for numerical
        and categorical features.

        Returns
        -------
        ColumnTransformer
        Preprocessing pipeline.
        """

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numeric_pipeline,
                    self.config.numerical_columns,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.config.categorical_columns,
                ),
            ]
        )

        return preprocessor

    def initiate_data_transformation(self) -> tuple[Path, Path, Path]:
        """
        Execute the complete data transformation workflow.

        Returns
        -------
        tuple
        Paths to the transformed train data,
        transformed test data,
        and saved preprocessing object.
        """
        try:

            logger.info("Starting data transformation.")

            # Load datasets
            train_df, test_df = self.load_data()

            # Convert configured object columns to numeric
            train_df = self.clean_numeric_columns(train_df)
            test_df = self.clean_numeric_columns(test_df)

            # Split features and target
            X_train, y_train = self.split_features_target(train_df)
            X_test, y_test = self.split_features_target(test_df)

            # Encode target
            y_train = y_train.map({"No": 0, "Yes": 1}).astype(np.int64)
            y_test = y_test.map({"No": 0, "Yes": 1}).astype(np.int64)

            # Create preprocessor
            preprocessor = self.get_preprocessor()

            logger.info("Fitting preprocessing pipeline on training data.")

            X_train_transformed = preprocessor.fit_transform(X_train)

            logger.info("Transforming testing data.")

            X_test_transformed = preprocessor.transform(X_test)

            logger.info(f"Target dtype: {y_train.dtype}")
            logger.info(f"Unique target values: {y_train.unique()}")

            logger.info(f"X_train_transformed dtype: {X_train_transformed.dtype}")

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]

            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            logger.info("Saving preprocessing object.")

            save_object(
                file_path=self.config.preprocessor_object_path, obj=preprocessor
            )

            logger.info("Saving transformed training array.")

            np.save(self.config.transformed_train_path, train_arr)

            logger.info("Saving transformed testing array.")

            np.save(self.config.transformed_test_path, test_arr)

            logger.info("Data transformation completed successfully.")

            return (
                self.config.transformed_train_path,
                self.config.transformed_test_path,
                self.config.preprocessor_object_path,
            )

        except Exception as e:
            logger.exception("Error occurred during data transformation.")
            raise CustomException(e, sys)
