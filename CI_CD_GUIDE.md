# 🚀 CI/CD Guide - GitHub Actions for ML Pipeline

## Overview

This project includes comprehensive GitHub Actions workflows for automated ML experimentation, testing, and deployment with DVC and MLFlow integration.

## 📋 Available Workflows

### 1. **CI - Tests and Code Quality** (`.github/workflows/ci.yml`)

**Triggers:** Push to main/develop, Pull Requests
**Purpose:** Automated testing, code quality, and basic functionality checks

**Features:**
- ✅ Python dependencies installation
- ✅ Code quality (flake8, mypy)
- ✅ Unit tests with coverage
- ✅ DVC functionality testing
- ✅ MLFlow basic operations
- ✅ Coverage reports to Codecov

### 2. **ML Experiment Pipeline** (`.github/workflows/experiment-pipeline.yml`)

**Triggers:** Code changes in `src/`, config changes, manual dispatch
**Purpose:** Full ML experiment execution in CI environment

**Features:**
- 🔄 Complete data → features → train → evaluate pipeline
- 🎯 Synthetic data generation for CI
- 📊 MLFlow experiment tracking
- 📈 Automatic experiment reports
- 💬 PR comments with results
- 📦 Artifact storage (models, metrics, plots)

**Manual Trigger:**
```bash
# Via GitHub UI: Actions → ML Experiment Pipeline → Run workflow
# Specify custom config file and experiment name
```

### 3. **MLFlow Model Registry** (`.github/workflows/mlflow-model-registry.yml`)

**Triggers:** Manual workflow dispatch
**Purpose:** Model promotion and deployment management

**Features:**
- 🏷️ Model registration and versioning
- ⬆️ Stage promotion (Staging → Production)
- ✅ Production model validation
- 📋 Deployment tracking
- 🔔 Deployment notifications

**Usage:**
```bash
# Via GitHub UI: Actions → MLFlow Model Registry → Run workflow
# Inputs:
# - model_name: "spillover_classifier"
# - stage: "Production"
# - run_id: (optional specific run)
```

### 4. **Scheduled Experiments** (`.github/workflows/scheduled-experiments.yml`)

**Triggers:** Daily at 2 AM UTC, manual dispatch
**Purpose:** Automated retraining and performance monitoring

**Features:**
- 🕐 Daily automated experiments
- 📊 Performance trend analysis
- 🚨 Automatic issue creation for problems
- 📈 7-day performance tracking
- 🔄 Fresh data simulation

## 🛠️ Setup Instructions

### 1. Repository Setup

```bash
# Initialize Git (if not done)
make git-setup

# Create first commit with all files
make git-commit

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/info_spillover.git
git push -u origin main
```

### 2. GitHub Repository Configuration

**Secrets (if needed for external services):**
- Go to Settings → Secrets and variables → Actions
- Add secrets like `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, etc.

**Branch Protection:**
- Settings → Branches → Add rule for `main`
- Enable "Require status checks" and select CI workflow

### 3. Workflow Permissions

Ensure GitHub Actions has proper permissions:
- Settings → Actions → General
- Workflow permissions: "Read and write permissions"
- Allow creating issues: ✅

## 🎯 Usage Examples

### Running Full Experiment Pipeline

```bash
# Trigger manually with custom config
# 1. Go to Actions → ML Experiment Pipeline
# 2. Click "Run workflow"
# 3. Specify config: experiments/configs/config.yaml
# 4. Add experiment name (optional)
```

### Model Deployment Workflow

```bash
# 1. Train model locally or via CI
# 2. Note the MLFlow run_id from UI
# 3. Go to Actions → MLFlow Model Registry
# 4. Run with: model_name="spillover_classifier", stage="Staging"
# 5. After validation, promote to "Production"
```

### Monitoring Workflow Results

```bash
# View in GitHub:
# 1. Actions tab → Select workflow run
# 2. Download artifacts (models, reports, databases)
# 3. Check Issues for automated alerts

# View MLFlow results:
# 1. Download mlflow database artifact
# 2. Place in project root
# 3. make mlflow-start
# 4. Browse to localhost:5000
```

## 📊 Workflow Outputs

### Artifacts Generated

**CI Workflow:**
- `test-results` - Coverage reports and test outputs
- `htmlcov/` - HTML coverage report

**Experiment Pipeline:**
- `experiment-results-{run_number}` - Complete experiment outputs
- `mlflow-database-{run_number}` - MLFlow tracking database
- `experiment_report.json` - Structured results

**Model Registry:**
- `deployment-summary-{run_number}` - Deployment details
- Updated MLFlow model registry

### Automatic Issue Creation

**Performance Alerts:**
- Created when model performance declines
- Daily monitoring via scheduled workflow
- Tagged: `performance`, `needs-attention`

**Deployment Notifications:**
- Production deployments create tracking issues
- Tagged: `deployment`, `production`

**Error Alerts:**
- Failed experiments create bug reports
- Tagged: `bug`, `scheduled`

## 🔧 Customization

### Adding New Experiment Configs

1. Create new config in `experiments/configs/`
2. Use in manual workflow dispatch
3. Workflow will automatically use the new config

### Custom Data Sources

Edit `experiment-pipeline.yml`:
```yaml
# Replace synthetic data generation with real data loading
- name: Load real data
  run: |
    # Your data loading logic
    python scripts/load_production_data.py
```

### Notification Integrations

Add Slack/Teams notifications:
```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🐛 Troubleshooting

### Common Issues

**1. MLFlow Database Conflicts**
```bash
# Clean slate in CI
export MLFLOW_TRACKING_URI=sqlite:///fresh_mlflow.db
```

**2. DVC Remote Access**
```bash
# Configure temporary remote for CI
dvc remote add -d ci_remote /tmp/dvc_storage
```

**3. Memory Issues**
```bash
# Reduce data size for CI
n_samples = 500  # Instead of 2000
```

### Debugging Workflows

**Enable Debug Logging:**
```yaml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

**Check Artifacts:**
- Download workflow artifacts
- Inspect logs and outputs locally

## 🚦 Status Badges

Add to README.md:
```markdown
![CI Status](https://github.com/YOUR_USERNAME/info_spillover/workflows/CI%20-%20Tests%20and%20Code%20Quality/badge.svg)
![Experiment Pipeline](https://github.com/YOUR_USERNAME/info_spillover/workflows/ML%20Experiment%20Pipeline/badge.svg)
```

## 📈 Performance Monitoring

The scheduled workflow automatically:
- Tracks model performance over time
- Compares against 7-day averages
- Creates alerts for declining performance
- Generates trend reports

Monitor via:
- GitHub Issues (automatic alerts)
- Workflow artifacts (detailed reports)
- MLFlow UI (complete experiment history)