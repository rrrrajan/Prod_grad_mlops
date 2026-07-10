from pathlib import Path
import sys
import json
import yaml
import logging
import joblib

from typing import Any

from src.exception import CustomException

logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path) -> dict[str, Any]:
    """
    Read a YAML file and return its contents.

    Parameters
    ----------
    path_to_yaml : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: list, verbose: bool = True) -> None:
    """
    Create one or more directories.

    Parameters
    ----------
    path_to_directories : list
        List of directory paths.

    verbose : bool, default=True
        Whether to log directory creation.
    """

    try:
        for path in path_to_directories:
            Path(path).mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info("Created directory at: %s", path)

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: Path, obj: Any) -> None:
    """
    Serialize and save a Python object using joblib.

    Parameters
    ----------
    file_path : Path
        Destination file path.

    obj : Any
        Python object to serialize.
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(obj, file_path)

        logger.info("Object saved successfully at: %s", file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: Path) -> Any:
    """
    Load a serialized Python object.

    Parameters
    ----------
    file_path : Path
        Path to the serialized object.

    Returns
    -------
    Any
        Loaded Python object.
    """

    try:
        logger.info("Loading object from: %s", file_path)

        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: Path, data: dict) -> None:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    file_path : Path
        Destination JSON file path.

    data : dict
        Dictionary to save.
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        logger.info("JSON file saved successfully at: %s", file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: Path) -> dict:
    """
    Load a JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents.
    """

    try:
        logger.info("Loading JSON file from: %s", file_path)

        with open(file_path, "r") as json_file:
            return json.load(json_file)

    except Exception as e:
        raise CustomException(e, sys)