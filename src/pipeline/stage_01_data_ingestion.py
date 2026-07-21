from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.initiate_data_ingestion()
