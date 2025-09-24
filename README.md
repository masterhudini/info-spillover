# Information Spillover Experiments

A comprehensive ML experimentation framework with DVC and MLFlow integration for information spillover analysis.

## Project Structure

```
info_spillover/
├── data/
│   ├── raw/                    # Raw, immutable data
│   ├── processed/              # Cleaned and processed data
│   ├── external/               # External data sources
│   └── interim/                # Intermediate processed data
├── src/
│   ├── data/                   # Data processing scripts
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and evaluation
│   ├── visualization/          # Visualization utilities
│   └── utils/                  # Common utilities
├── experiments/
│   ├── configs/                # Experiment configurations
│   ├── outputs/                # Experiment outputs
│   └── logs/                   # Experiment logs
├── models/
│   ├── saved/                  # Trained models
│   └── checkpoints/            # Model checkpoints
├── notebooks/
│   ├── exploratory/            # Exploratory analysis
│   └── reports/                # Report notebooks
├── scripts/                    # Utility scripts
├── tests/                      # Unit and integration tests
└── docs/                       # Documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd info_spillover

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or
pip install -r requirements.txt
pip install -e .
```

### 2. Google Cloud Setup (Required for BigQuery)

This project uses Google BigQuery for data storage and processing. You need to configure Google Cloud authentication:

#### Option A: Service Account (Recommended for production)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Go to IAM & Admin → Service Accounts
4. Create a new service account with BigQuery permissions:
   - `BigQuery Data Editor`
   - `BigQuery Job User`
   - `BigQuery User`
5. Download the JSON key file
6. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

#### Option B: User Authentication (For development)

```bash
# Install Google Cloud SDK
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

#### Verify Setup

```bash
# Test your Google Cloud configuration
python src/utils/gcp_setup.py

# Or run the verification directly
python -c "from src.utils.gcp_setup import GCPAuthenticator; GCPAuthenticator.test_bigquery_access()"
```

### 3. Project Initialization

```bash
# Setup project structure and initialize DVC
make setup

# Copy environment variables template
cp .env.example .env
# Edit .env with your specific settings
```

### 4. Run Hierarchical Sentiment Analysis

#### Quick Start - Full Pipeline

```bash
# Run the complete hierarchical analysis pipeline
python src/main_pipeline.py
```

This will execute:
1. 📊 Data processing and feature engineering
2. 🔄 Spillover analysis (Diebold-Yilmaz framework)
3. 🧠 Hierarchical modeling (LSTM + GNN)
4. 💼 Economic evaluation and backtesting
5. 📋 Comprehensive report generation

#### Individual Components

```bash
# Test data processing only
python src/data/hierarchical_data_processor.py

# Run spillover analysis
python src/analysis/diebold_yilmaz_spillover.py

# Test economic evaluation
python src/evaluation/economic_evaluation.py

# Start MLFlow UI to track experiments
make mlflow-start
# Visit http://localhost:5000
```

### 5. Configuration

Edit `experiments/configs/hierarchical_config.yaml` to customize:

- **Data settings**: Date ranges, preprocessing options
- **Model architecture**: LSTM vs Transformer, GNN type
- **Spillover analysis**: Forecast horizons, VAR parameters
- **Backtesting**: Transaction costs, risk parameters
- **Output**: Visualization and reporting options

## DVC Integration

DVC (Data Version Control) manages data pipelines and tracks experiments.

### Key Commands

```bash
# Reproduce entire pipeline
dvc repro

# Show pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Show plots
dvc plots show

# Add data to DVC tracking
dvc add data/raw/your_data.csv
```

### Pipeline Stages

The `dvc.yaml` defines the following stages:
1. **prepare_data**: Data cleaning and splitting
2. **feature_engineering**: Feature extraction and selection
3. **train**: Model training with hyperparameters
4. **evaluate**: Model evaluation on test set

## MLFlow Integration

MLFlow tracks experiments, parameters, metrics, and models.

### Features

- **Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Store plots, models, and other experiment outputs
- **UI Dashboard**: Web interface for experiment comparison

### Usage

```python
from src.utils.mlflow_utils import MLFlowTracker

# Initialize tracker
tracker = MLFlowTracker("my_experiment")

# Start run and log metrics
with tracker.start_run():
    tracker.log_params({"lr": 0.01, "epochs": 100})
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    tracker.log_model(model, "model")
```

## Configuration

All experiment configurations are stored in `experiments/configs/config.yaml`.

### Key Configuration Sections

- **Data**: Paths and preprocessing options
- **Model**: Model type and hyperparameters
- **Training**: Training configuration and validation
- **Features**: Feature engineering settings
- **MLFlow**: Tracking and logging configuration

### Example Usage

```bash
# Run with custom config
python scripts/run_experiment.py --config experiments/configs/my_config.yaml

# Run only training step
python scripts/run_experiment.py --only-train
```

## Development

### Code Quality

```bash
# Run tests
make test

# Code formatting
make format

# Linting
make lint
```

### Adding New Experiments

1. Create new config file in `experiments/configs/`
2. Modify scripts in `src/` as needed
3. Update `dvc.yaml` if pipeline changes
4. Run experiment: `python scripts/run_experiment.py --config your_config.yaml`

### Directory Structure Guidelines

- **data/raw/**: Never modify files here, treat as read-only
- **data/processed/**: Store cleaned, ready-to-use datasets
- **experiments/configs/**: Experiment configurations in YAML format
- **models/saved/**: Final trained models
- **notebooks/**: Use for exploration and reporting, not production code

## Environment Variables

Key environment variables (see `.env.example`):

- `MLFLOW_TRACKING_URI`: MLFlow tracking server URI
- `DVC_REMOTE_URL`: Remote storage for DVC (optional)
- `DATA_PATH`: Base path for data files
- `LOG_LEVEL`: Logging verbosity

## Makefile Commands

- `make install`: Install dependencies
- `make setup`: Initialize project structure
- `make experiment`: Run full pipeline
- `make mlflow-start`: Start MLFlow UI
- `make test`: Run tests
- `make clean`: Clean temporary files
- `make help`: Show all available commands

## Troubleshooting

### Common Issues

1. **DVC not initialized**: Run `make dvc-setup`
2. **MLFlow UI not starting**: Check if port 5000 is available
3. **Import errors**: Make sure you installed with `pip install -e .`
4. **Permission issues**: Check file permissions for scripts

### Getting Help

- Check the logs in `experiments/logs/`
- Review MLFlow UI for experiment details
- Use `dvc dag` to debug pipeline issues

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation
4. Use meaningful commit messages
5. Test with `make test` before committing

## License

[Specify your license here]