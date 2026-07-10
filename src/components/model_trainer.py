import json
import joblib
import numpy as np
import sys

from src.utils.common import save_json
from src.logger import logger
from src.exception import CustomException


from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.base import ClassifierMixin

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact


class ModelTrainer:
    """
    Component responsible for training, evaluating and selecting
    the best machine learning model.
    """

    _SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "adaboost": AdaBoostClassifier,
        "extra_trees": ExtraTreesClassifier,
    }


    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
        """
        Load transformed training and testing datasets.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test.
        """

        try:
            logger.info("Loading transformed datasets.")

            train_array = np.load(self.config.transformed_train_path)
            test_array = np.load(self.config.transformed_test_path)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logger.info(
                "Training data loaded successfully. X_train shape: %s, y_train shape: %s",
                X_train.shape,
                y_train.shape,
            )

            logger.info(
                "Testing data loaded successfully. X_test shape: %s, y_test shape: %s",
                X_test.shape,
                y_test.shape,
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.exception("Failed to load transformed datasets.")
            raise CustomException(e, sys)   

    def get_models(self) -> Dict[str, ClassifierMixin]:
        """
        Create and return all enabled machine learning models.

        Returns
        -------
        Dict[str, ClassifierMixin]
        Dictionary containing instantiated models.
        """

        try:
            logger.info("Initializing machine learning models.")

            models = {}

            configured_models = self.config.model_params

            for model_name, model_config in configured_models.items():

                if not model_config.get("enabled", False):
                    logger.info("Skipping disabled model: %s", model_name)
                    continue

                if model_name not in self._SUPPORTED_MODELS:
                    raise CustomException(
                        f"Unsupported model: {model_name}",
                        sys
                    )

                model_class = self._SUPPORTED_MODELS[model_name]

                model_params = {
                    key: value
                    for key, value in model_config.items()
                    if key != "enabled"
                }

                models[model_name] = model_class(**model_params)

                logger.info(
                    "Initialized model: %s with parameters: %s",
                    model_name,
                    model_params,
                )

            
            if not models:
                raise CustomException(
                    "No enabled models found in configuration.",
                    sys,
                )

            logger.info("Successfully initialized %d models.", len(models))

            return models

        except Exception as e:
            logger.exception("Failed to initialize models.")
            raise CustomException(e, sys)
        

    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
      y_test: np.ndarray, models: Dict[str, ClassifierMixin]) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all configured machine learning models.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.

        X_test : np.ndarray
            Testing feature matrix.

        y_train : np.ndarray
            Training target values.

        y_test : np.ndarray
            Testing target values.

        models : Dict[str, ClassifierMixin]
            Dictionary containing instantiated machine learning models.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing each trained model and its
            evaluation metrics.
        """

        try:

            logger.info("Starting model training and evaluation.")

            results: Dict[str, Dict[str, Any]] = {}

            for model_name, model in models.items():

                logger.info("Training model: %s", model_name)

                # Train model
                model.fit(X_train, y_train)

                logger.info("Generating predictions for model: %s", model_name)

                y_pred = model.predict(X_test)

                # Compute evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                # ROC-AUC
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]

                    roc_auc = roc_auc_score(
                        y_test,
                        y_score,
                    )
                else:
                    roc_auc = None

                results[model_name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                }

                logger.info(
                    "Completed evaluation for %s | "
                    "Accuracy: %.4f | "
                    "Precision: %.4f | "
                    "Recall: %.4f | "
                    "F1: %.4f | "
                    "ROC-AUC: %s",
                    model_name,
                    accuracy,
                    precision,
                    recall,
                    f1,
                    f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
                )

            logger.info(
                "Successfully trained and evaluated %d models.",
                len(results),
            )

            return results

        except Exception as e:

            logger.exception("Failed during model training and evaluation.")

            raise CustomException(e, sys)    
        

    def select_best_model(self, results: Dict[str, Dict[str, Any]]) -> tuple[str, ClassifierMixin, Dict[str, Any]]:
        """
        Select the best-performing model based on the configured metric.

        Parameters
        ----------
        results : Dict[str, Dict[str, Any]]
            Dictionary containing trained models and their evaluation metrics.

        Returns
        -------
        tuple[str, ClassifierMixin, Dict[str, Any]]
            A tuple containing:
            - Best model name
            - Best trained model
            - Best model evaluation metrics
        """

        try:

            logger.info("Selecting the best-performing model.")

            evaluation_metric = self.config.model_params.get(
                "evaluation_metric",
                "roc_auc",
            )

            logger.info(
                "Model evaluation metric: %s",
                evaluation_metric,
            )

            if not results:
                raise CustomException(
                    "No model evaluation results available.",
                    sys,
                )

            best_model_name = None
            best_model = None
            best_metrics = None

            best_score = float("-inf")

            for model_name, result in results.items():

                score = result.get(evaluation_metric)

                # Skip models without the configured metric
                if score is None:
                    logger.warning(
                        "Skipping model '%s' because '%s' is unavailable.",
                        model_name,
                        evaluation_metric,
                    )
                    continue

                logger.info(
                    "Model: %s | %s = %.4f",
                    model_name,
                    evaluation_metric,
                    score,
                )

                if score > best_score:

                    best_score = score
                    best_model_name = model_name
                    best_model = result["model"]

                    best_metrics = {
                        key: value
                        for key, value in result.items()
                        if key != "model"
                    }

            if best_model is None:
                raise CustomException(
                    f"No model contains the metric '{evaluation_metric}'.",
                    sys,
                )

            logger.info(
                "Best model selected: %s (%.4f)",
                best_model_name,
                best_score,
            )

            return (
                best_model_name,
                best_model,
                best_metrics,
            )

        except Exception as e:

            logger.exception("Failed to select the best model.")

            raise CustomException(e, sys)    
        

    def save_model(self, model) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        model : sklearn estimator
            Trained machine learning model.
        """

        try:

            model_path = Path(self.config.trained_model_path)

            model_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, model_path)

            logger.info(f"Model saved at {model_path}")

        except Exception as e:
            raise CustomException(e, sys)
    

    def save_metrics(self, metrics: dict) -> None:
        """
        Save evaluation metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary containing evaluation metrics.
        """

        try:

            metrics_path = Path(self.config.metrics_file_name)

            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            save_json(
                file_path=metrics_path,
                data=metrics
            )

            logger.info(f"Metrics saved at {metrics_path}")

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Execute the complete Model Trainer workflow.

        Returns
        -------
        ModelTrainerArtifact
        """

        try:
            logger.info("Starting Model Trainer stage.")

            # Load transformed datasets
            X_train, X_test, y_train, y_test = self.load_data()

            # Initialize models
            models = self.get_models()

            # Train and evaluate
            results = self.train_and_evaluate(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
            )

            # Select best model
            (
                best_model_name,
                best_model,
                best_metrics,
            ) = self.select_best_model(results)

            logger.info("Best model selected: %s", best_model_name)

            # Save trained model
            self.save_model(best_model)

            # Save metrics
            self.save_metrics(best_metrics)

            # Prepare model report
            model_report = {}

            for model_name, result in results.items():

                model_report[model_name] = {
                    key: value
                    for key, value in result.items()
                    if key != "model"
                }

            save_json(
                file_path=self.config.model_report_file_name,
                data=model_report,
            )

            logger.info("Model report saved successfully.")

            artifact = ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                metrics_file_path=self.config.metrics_file_name,
                model_report_file_name=self.config.model_report_file_name,
                best_model_name=best_model_name,
            )

            logger.info("Model Trainer stage completed successfully.")

            return artifact

        except Exception as e:
            logger.exception("Model Trainer stage failed.")
            raise CustomException(e, sys)