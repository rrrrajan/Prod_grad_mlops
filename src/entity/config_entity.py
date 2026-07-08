from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    """

    root_dir: Path
    raw_data_path: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the Data Validation component.
    """

    root_dir: Path
    STATUS_FILE: Path
    unzip_data_dir: Path
    all_schema: dict