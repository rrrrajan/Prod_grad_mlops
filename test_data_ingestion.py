from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

config_manager = ConfigurationManager()

ingestion_config = config_manager.get_data_ingestion_config()

data_ingestion = DataIngestion(ingestion_config)

data_ingestion.initiate_data_ingestion()

print("Method executed successfully!")