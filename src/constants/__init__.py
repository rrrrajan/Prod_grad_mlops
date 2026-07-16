from pathlib import Path

# ==============================================================================
# Project Root
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# ==============================================================================
# Project Directories
# ==============================================================================

CONFIG_DIR = ROOT_DIR / "config"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODELS_DIR = ROOT_DIR / "models"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

RESULTS_DIR = ROOT_DIR / "results"

# ==============================================================================
# Configuration Files
# ==============================================================================

CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"

PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"

SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"
