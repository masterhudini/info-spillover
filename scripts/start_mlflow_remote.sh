#!/bin/bash

# Start MLFlow UI server accessible from remote machines
echo "Starting MLFlow tracking server for remote access..."

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

# Get host IP address
HOST_IP=$(hostname -I | awk '{print $1}')

echo "Configuration:"
echo "  Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "  Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "  Tracking URI: $MLFLOW_TRACKING_URI"
echo "  Host IP: $HOST_IP"
echo ""

echo "⚠️  SECURITY WARNING:"
echo "   This will make MLFlow accessible from any network interface!"
echo "   Make sure your firewall is configured properly."
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Check if MLFlow is already running on port 5000
if lsof -i :5000 >/dev/null 2>&1; then
    echo "Port 5000 is already in use. Stopping existing process..."
    pkill -f mlflow
    sleep 2
fi

# Start the MLFlow server accessible from all interfaces
echo "Starting MLFlow server..."
mlflow ui \
  --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
  --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts &

# Get the process ID
MLFLOW_PID=$!

# Wait a moment to check if it started successfully
sleep 3

if ps -p $MLFLOW_PID > /dev/null; then
    echo "✅ MLFlow UI started successfully!"
    echo ""
    echo "Access URLs:"
    echo "  Local: http://localhost:5000"
    echo "  Remote: http://$HOST_IP:5000"
    echo ""
    echo "From Windows:"
    echo "  1. SSH tunnel: ssh -L 5000:localhost:5000 $USER@$HOST_IP"
    echo "     Then browse: http://localhost:5000"
    echo ""
    echo "  2. Direct access: http://$HOST_IP:5000"
    echo "     (requires firewall port 5000 open)"
    echo ""
    echo "To stop: kill $MLFLOW_PID or Ctrl+C"

    # Save PID for later reference
    echo $MLFLOW_PID > mlflow.pid

    # Wait for the process to finish
    wait $MLFLOW_PID
else
    echo "❌ Failed to start MLFlow UI"
    exit 1
fi