from src.config.configuration import ConfigurationManager
from src.components.deployment import Deployment


def test_download_registered_model():

    config = ConfigurationManager()

    deployment = Deployment(
        deployment_config=config.get_deployment_config(),
        mlflow_config=config.get_mlflow_config(),
    )

    print("Starting deployment test...")

    deployment.connect_to_mlflow()
    print("Connected.")

    download_info = deployment.download_registered_model()

    print(f"Version      : {download_info['version']}")
    print(f"Downloaded to: {download_info['path']}")

if __name__ == "__main__":
    test_download_registered_model()