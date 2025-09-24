#!/bin/bash

# Start MLFlow UI server
echo "Starting MLFlow tracking server..."

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Set the backend store URI (SQLite database)
export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"

# Set the default artifact root
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# Set tracking URI for consistency
export MLFLOW_TRACKING_URI="file:./mlruns"

echo "Configuration:"
echo "  Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "  Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "  Tracking URI: $MLFLOW_TRACKING_URI"
echo ""

# Start the MLFlow server
mlflow ui \
  --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
  --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts

echo "MLFlow UI available at http://localhost:5000"