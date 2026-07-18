import json

import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.experiment_tracking.experiment_tracker import ExperimentTracker
from src.exception import CustomException
from src.logger import logger


class ModelEvaluation:
    """
    Component responsible for evaluating the trained model on
    the transformed test dataset and generating evaluation
    artifacts.
    """

    def __init__(self, config: ModelEvaluationConfig, tracker: ExperimentTracker):
        """
        Initialize the ModelEvaluation component.

        Parameters
        ----------
        config : ModelEvaluationConfig
            Configuration object for the evaluation stage.
        """
        self.config = config
        self.tracker = tracker

    def load_model(self):
        """
        Load the trained model from disk.

        Returns
        -------
        object
            Trained machine learning model.
        """

        try:
            logger.info("Loading trained model from %s", self.config.trained_model_path)

            model = joblib.load(self.config.trained_model_path)

            logger.info("Trained model loaded successfully.")

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def load_test_data(self):
        """
        Load the transformed test dataset.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Feature matrix (X_test) and target vector (y_test).
        """

        try:
            logger.info(
                "Loading transformed test data from %s", self.config.test_array_path
            )

            test_array = np.load(self.config.test_array_path)

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            logger.info("Test dataset loaded successfully. Shape: %s", test_array.shape)

            return X_test, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, object]:
        """
        Evaluate the trained model on the test dataset.

        Parameters
        ----------
        model : object
            Trained machine learning model.

        X_test : np.ndarray
            Test feature matrix.

        y_test : np.ndarray
            Ground truth labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics and
            generated evaluation artifacts.
        """

        try:
            logger.info("Starting model evaluation.")

            y_pred = model.predict(X_test)

            y_prob = None
            roc_auc = None

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "roc_auc": roc_auc,
            }

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            )

            cm = confusion_matrix(
                y_test,
                y_pred,
            )

            logger.info("Model evaluation completed successfully.")

            return {
                "metrics": metrics,
                "classification_report": report,
                "confusion_matrix": cm,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def save_metrics(self, metrics: dict) -> None:
        """
        Save evaluation metrics as a JSON file.

        Parameters
        ----------
        metrics : dict
            Dictionary containing evaluation metrics.
        """

        try:
            logger.info(
                "Saving evaluation metrics to %s", self.config.metrics_file_name
            )

            with open(
                self.config.metrics_file_name,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=4)

            logger.info("Evaluation metrics saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def save_metadata(self, model) -> None:
        """
        Save evaluation metadata as a JSON file.
        """

        try:
            logger.info(
                "Saving evaluation metadata to %s",
                self.config.metadata_file_name,
            )

            metadata = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "evaluation_metric": self.config.evaluation_metric,
                "experiment_name": self.config.experiment_name,
                "run_name": self.config.run_name,
                "log_model": self.config.log_model,
                "register_model": self.config.register_model,
            }

            with open(
                self.config.metadata_file_name,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4,
                )

            logger.info("Evaluation metadata saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def save_classification_report(self, report: dict) -> None:
        """
        Save the classification report as JSON.

        Parameters
        ----------
        report : dict
            Classification report generated by sklearn.
        """

        try:
            logger.info(
                "Saving classification report to %s",
                self.config.classification_report_file_name,
            )

            with open(
                self.config.classification_report_file_name,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(report, f, indent=4)

            logger.info("Classification report saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def save_confusion_matrix(self, confusion_matrix_data: np.ndarray) -> None:
        """
        Save the confusion matrix as a JSON file.

        Parameters
        ----------
        confusion_matrix_data : np.ndarray
            Confusion matrix generated during evaluation.
        """

        try:
            logger.info(
                "Saving confusion matrix to %s",
                self.config.confusion_matrix_json_file_name,
            )

            with open(
                self.config.confusion_matrix_json_file_name,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    confusion_matrix_data.tolist(),
                    file,
                    indent=4,
                )

            logger.info("Confusion matrix  JSON saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def plot_confusion_matrix(self, confusion_matrix_data: np.ndarray) -> None:
        """
        Generate and save the confusion matrix plot.

        Parameters
        ----------
        confusion_matrix_data : np.ndarray
            Confusion matrix.
        """

        try:
            logger.info("Generating confusion matrix plot.")

            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data)

            disp.plot()

            plt.tight_layout()

            plt.savefig(
                self.config.confusion_matrix_file_name,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            logger.info("Confusion matrix plot saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def plot_roc_curve(self, model, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Generate and save the ROC curve.

        Parameters
        ----------
        model : object
            Trained machine learning model.

        X_test : np.ndarray
            Test feature matrix.

        y_test : np.ndarray
            True labels.
        """

        try:
            if not hasattr(model, "predict_proba"):
                logger.warning(
                    "ROC curve skipped because the model "
                    "does not support predict_proba()."
                )
                return

            logger.info("Generating ROC curve.")

            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test,
            )

            plt.tight_layout()

            plt.savefig(
                self.config.roc_curve_file_name,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            logger.info("ROC curve plot saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def log_experiment(self, model, metrics: dict, X_test: np.ndarray) -> None:
        """
        Log evaluation metrics, artifacts, and model using the
        configured experiment tracker.
        """

        try:
            logger.info("Logging evaluation results to experiment tracker.")

            with self.tracker.run(run_name=self.config.run_name):

                # -----------------------------
                # Tags
                # -----------------------------
                self.tracker.set_tags(
                    {
                        "stage": "model_evaluation",
                        "model_type": type(model).__name__,
                    }
                )

                # -----------------------------
                # Metrics
                # -----------------------------
                self.tracker.log_metrics(metrics)

                # -----------------------------
                # Evaluation parameters
                # -----------------------------
                self.tracker.log_params(
                    {
                        "evaluation_metric": self.config.evaluation_metric,
                        "test_samples": len(X_test),
                        "n_features": X_test.shape[1],
                    }
                )

                # -----------------------------
                # Model parameters
                # -----------------------------
                if hasattr(model, "get_params"):
                    self.tracker.log_params(model.get_params())

                # -----------------------------
                # Evaluation artifacts
                # -----------------------------
                artifact_paths = [
                    self.config.metrics_file_name,
                    self.config.metadata_file_name,
                    self.config.classification_report_file_name,
                    self.config.confusion_matrix_json_file_name,
                    self.config.confusion_matrix_file_name,
                    self.config.roc_curve_file_name,
                ]

                self.tracker.log_artifacts(artifact_paths)

                # -----------------------------
                # Trained model
                # -----------------------------
                if self.config.log_model:

                    registered_model_name = None

                    if self.config.register_model:
                        registered_model_name = self.config.registered_model_name

                        logger.info("Model will be registered as '%s'.",
                            registered_model_name,
                        )

                    self.tracker.log_model(
                        model=model,
                        artifact_path="model",
                        registered_model_name=registered_model_name,
                    )

            logger.info("Experiment tracking completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> dict:
        """
        Execute the complete Model Evaluation pipeline.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """

        try:
            logger.info("========== Model Evaluation Started ==========")

            # Load trained model
            model = self.load_model()

            # Load transformed test data
            X_test, y_test = self.load_test_data()

            # Evaluate model
            evaluation_results = self.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            # Save evaluation metrics
            self.save_metrics(evaluation_results["metrics"])

            # Save confusion matrix values
            self.save_confusion_matrix(evaluation_results["confusion_matrix"])

            # Save evaluation metadata
            self.save_metadata(model)

            # Save classification report
            self.save_classification_report(evaluation_results["classification_report"])

            # Generate confusion matrix
            self.plot_confusion_matrix(evaluation_results["confusion_matrix"])

            # Generate ROC curve
            self.plot_roc_curve(
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            # Log results to MLflow
            self.log_experiment(
                model=model,
                metrics=evaluation_results["metrics"],
                X_test=X_test,
            )

            logger.info("========== Model Evaluation Completed Successfully ==========")

            return evaluation_results

        except Exception as e:
            logger.exception("Error occurred during Model Evaluation.")
            raise CustomException(e, sys)
