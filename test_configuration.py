from src.config.configuration import ConfigurationManager

config = ConfigurationManager()

ingestion = config.get_data_ingestion_config()

print(ingestion)
print(ingestion.root_dir)
print(ingestion.raw_data_path)
print(ingestion.local_data_file)
