"""
Inference Pipeline

Responsible for serving predictions.

Currently acts as a placeholder.

Future responsibilities:
- Load production model
- Preprocess incoming data
- Generate predictions
"""

from src.logger import logger


class InferencePipeline:
    """
    Handles model inference.
    """

    def __init__(self):
        logger.info("Initializing Inference Pipeline...")

    def predict(self, data):
        logger.info("Inference request received.")

        raise NotImplementedError("Inference Pipeline is not implemented yet.")
