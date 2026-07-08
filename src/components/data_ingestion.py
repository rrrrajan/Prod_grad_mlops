from pathlib import Path
import shutil

from src.entity.config_entity import DataIngestionConfig
from src.logger import logger


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        source_path = self.config.raw_data_path
        destination_path = self.config.local_data_file

        logger.info(f"Source file      : {source_path}")
        logger.info(f"Destination file : {destination_path}")

        shutil.copy(source_path, destination_path)

        logger.info("Dataset copied successfully.")