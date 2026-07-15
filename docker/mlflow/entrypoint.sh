#!/bin/sh

set -e

echo "Starting MLflow Tracking Server..."

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}"