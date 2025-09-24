#!/bin/bash

# Start MLFlow UI server in background
echo "Starting MLFlow tracking server in background..."

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Set the backend store URI (SQLite database)
export MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow.db"

# Set the default artifact root
export MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# Set tracking URI for consistency
export MLFLOW_TRACKING_URI="file:./mlruns"

# Suppress git warnings
export GIT_PYTHON_REFRESH=quiet

# Check if MLFlow is already running on port 5000
if lsof -i :5000 >/dev/null 2>&1; then
    echo "MLFlow is already running on port 5000"
    echo "Visit http://localhost:5000 to access the UI"
    exit 0
fi

echo "Configuration:"
echo "  Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "  Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "  Tracking URI: $MLFLOW_TRACKING_URI"
echo ""

# Start the MLFlow server in background
nohup mlflow ui \
  --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
  --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts > mlflow.log 2>&1 &

# Get the process ID
MLFLOW_PID=$!

# Wait a moment to check if it started successfully
sleep 3

if ps -p $MLFLOW_PID > /dev/null; then
    echo "✅ MLFlow UI started successfully!"
    echo "   Process ID: $MLFLOW_PID"
    echo "   Log file: mlflow.log"
    echo "   URL: http://localhost:5000"
    echo ""
    echo "To stop MLFlow:"
    echo "   kill $MLFLOW_PID"
    echo "   or use: pkill -f mlflow"

    # Save PID for later reference
    echo $MLFLOW_PID > mlflow.pid
else
    echo "❌ Failed to start MLFlow UI"
    echo "Check mlflow.log for details"
    exit 1
fi