import logging
import os
from datetime import datetime

from src.constants import ARTIFACTS_DIR

# =============================================================================
# Log Directory
# =============================================================================

RUN_TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Allow overriding the log directory inside containers.
LOGS_DIR = os.getenv(
    "LOG_DIR",
    str(ARTIFACTS_DIR / "logs"),
)

RUN_LOG_DIR = os.path.join(LOGS_DIR, RUN_TIMESTAMP)

handlers = [
    logging.StreamHandler(),
]

# =============================================================================
# File Logging (if writable)
# =============================================================================

try:
    os.makedirs(RUN_LOG_DIR, exist_ok=True)

    LOG_FILE_PATH = os.path.join(
        RUN_LOG_DIR,
        "running.log",
    )

    handlers.append(logging.FileHandler(LOG_FILE_PATH))

except OSError:
    # File logging unavailable (e.g. read-only filesystem or permissions).
    pass

# =============================================================================
# Configure Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger("customer_churn_mlops")
