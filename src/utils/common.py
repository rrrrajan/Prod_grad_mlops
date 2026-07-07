from pathlib import Path
import sys

import yaml

from src.exception import CustomException

def read_yaml(path_to_yaml: Path):
    """
    Reads a YAML file.
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content

    except Exception as e:
        raise CustomException(e, sys)