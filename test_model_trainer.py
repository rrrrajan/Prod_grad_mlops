from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer


def main():
    config = ConfigurationManager()

    model_trainer_config = config.get_model_trainer_config()

    model_trainer = ModelTrainer(config=model_trainer_config)

    artifact = model_trainer.initiate_model_trainer()

    print("\n===== Model Trainer Artifact =====")
    print(artifact)


if __name__ == "__main__":
    main()
