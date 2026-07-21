from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager

config = ConfigurationManager()

data_validation_config = config.get_data_validation_config()

data_validation = DataValidation(config=data_validation_config)

status = data_validation.validate_all_columns()

print(f"Validation Status : {status}")
