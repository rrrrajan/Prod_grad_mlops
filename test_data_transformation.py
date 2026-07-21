from src.components.data_transformation import DataTransformation
from src.config.configuration import ConfigurationManager

config = ConfigurationManager()

data_transformation_config = config.get_data_transformation_config()

data_transformation = DataTransformation(config=data_transformation_config)

train_path, test_path, preprocessor_path = (
    data_transformation.initiate_data_transformation()
)

print(train_path)
print(test_path)
print(preprocessor_path)
