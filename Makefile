# Makefile for info_spillover project

.PHONY: install install-dev setup clean test lint format dvc-setup mlflow-start experiment help

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install project dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Setup project (DVC, MLFlow, etc.)"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code"
	@echo "  dvc-setup    - Initialize DVC"
	@echo "  mlflow-start - Start MLFlow UI"
	@echo "  experiment   - Run full experiment pipeline"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .

# Project setup
setup: dvc-setup
	@echo "Setting up project structure..."
	mkdir -p data/{raw,processed,external,interim}
	mkdir -p models/{saved,checkpoints}
	mkdir -p experiments/{outputs,logs}
	mkdir -p notebooks/{exploratory,reports}
	@echo "Project setup complete!"

# DVC setup
dvc-setup:
	@echo "Initializing DVC..."
	dvc init --no-scm
	dvc remote add -d gcs_bucket /home/Hudini/gcs
	@echo "DVC initialized with GCS remote!"

test-dvc:
	python scripts/test_dvc.py

# MLFlow
setup-mlflow:
	python scripts/setup_mlflow.py

mlflow-start:
	@echo "Starting MLFlow UI..."
	./scripts/start_mlflow.sh

mlflow-bg:
	@echo "Starting MLFlow UI in background..."
	./scripts/start_mlflow_background.sh

mlflow-remote:
	@echo "Starting MLFlow UI for remote access..."
	./scripts/start_mlflow_remote.sh

mlflow-stop:
	@echo "Stopping MLFlow UI..."
	@if [ -f mlflow.pid ]; then \
		kill `cat mlflow.pid` && rm mlflow.pid && echo "MLFlow stopped"; \
	else \
		pkill -f mlflow && echo "MLFlow stopped"; \
	fi

show-host-ip:
	@echo "Internal IP: $(shell hostname -I | awk '{print $$1}')"
	@echo "External IP: 34.118.75.91"
	@echo ""
	@echo "Windows SSH tunnel command (external):"
	@echo "ssh -L 5000:localhost:5000 $(USER)@34.118.75.91"

test-mlflow:
	python -c "import mlflow; print('MLFlow version:', mlflow.version.VERSION); mlflow.set_tracking_uri('file:./mlruns'); print('Tracking URI:', mlflow.get_tracking_uri())"

# Data and experiment pipeline
setup-data:
	python scripts/setup_data.py

experiment:
	python scripts/run_experiment.py

sample-experiment:
	export GIT_PYTHON_REFRESH=quiet && python examples/sample_experiment.py

experiment-config:
	python scripts/run_experiment.py --config experiments/configs/config.yaml

# Data pipeline steps
prepare-data:
	python src/data/prepare_data.py

build-features:
	python src/features/build_features.py

train:
	python src/models/train_model.py --config experiments/configs/config.yaml

evaluate:
	python src/models/evaluate_model.py

# DVC pipeline
dvc-repro:
	dvc repro

dvc-dag:
	dvc dag

dvc-metrics:
	dvc metrics show

dvc-plots:
	dvc plots show

# Code quality
test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Git and versioning
git-setup:
	git init
	git branch -m main
	git add .gitignore README.md
	git commit -m "Initial commit with project structure 🚀\n\nGenerated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

git-commit:
	git add .
	git commit -m "Update project with MLFlow and DVC integration 📊\n\n- Complete ML experiment pipeline\n- GitHub Actions CI/CD\n- MLFlow tracking and model registry\n- DVC data versioning\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

# Environment
env-export:
	pip freeze > requirements_frozen.txt

# Documentation
docs-build:
	@echo "Building documentation..."
	# Add documentation build commands here

# Development workflow
dev-setup: install-dev setup
	@echo "Development environment ready!"

full-clean: clean
	rm -rf .dvc/cache
	rm -rf mlruns/