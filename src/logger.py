import logging
from datetime import datetime
from pathlib import Path

from src.constants import ARTIFACTS_DIR

# Create logs directory
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Create a unique folder for each run
RUN_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_LOG_DIR = LOGS_DIR / RUN_TIMESTAMP

# Create directories if they don't exist
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE_PATH = RUN_LOG_DIR / "running.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Project logger
logger = logging.getLogger("customer_churn_mlops")