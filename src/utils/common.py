from pathlib import Path
import sys
import yaml
import logging

import pickle
from typing import Any

from src.exception import CustomException

logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path):
    try:
        with open(path_to_yaml, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: list, verbose=True):
    try:
        for path in path_to_directories:
            Path(path).mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

def save_object(path: Path, obj: Any) -> None:
    """
    Save a Python object to disk using pickle.

    Parameters
    ----------
    path : Path
        Destination path where the object will be saved.

    obj : Any
        Python object to serialize.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(obj, file)

        logger.info(f"Object saved successfully at: {path}")

    except Exception as e:
        raise CustomException(e, sys)    