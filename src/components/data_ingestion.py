import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logger


class DataIngestion:
    """
    Responsible for ingesting the raw dataset and preparing
    train and test datasets for downstream pipeline stages.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the data ingestion workflow.

        Returns
        -------
        DataIngestionArtifact
            Artifact containing paths to the generated train and test datasets.
        """

        try:
            logger.info("Starting data ingestion.")

            # Copy raw dataset to artifacts
            logger.info(
                "Copying raw dataset from %s to %s",
                self.config.raw_data_path,
                self.config.local_data_path,
            )

            shutil.copy(
                self.config.raw_data_path,
                self.config.local_data_path,
            )

            logger.info("Raw dataset copied successfully.")

            # Read dataset
            logger.info("Loading dataset into DataFrame.")

            df = pd.read_csv(self.config.local_data_path)

            logger.info("Dataset loaded successfully with shape %s", df.shape)

            # Split dataset
            logger.info(
                "Splitting dataset into train and test sets "
                "(test_size=%s, random_state=%s).",
                self.config.test_size,
                self.config.random_state,
            )

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[self.config.target_column],
            )

            logger.info("Training dataset shape: %s", train_df.shape)
            logger.info("Testing dataset shape: %s", test_df.shape)

            # Save train dataset
            train_df.to_csv(
                self.config.train_data_path,
                index=False,
            )

            # Save test dataset
            test_df.to_csv(
                self.config.test_data_path,
                index=False,
            )

            logger.info("Train and test datasets saved successfully.")

            return DataIngestionArtifact(
                train_file_path=self.config.train_data_path,
                test_file_path=self.config.test_data_path,
            )

        except Exception as e:
            logger.exception("Data ingestion failed.")
            raise CustomException(e, sys)
