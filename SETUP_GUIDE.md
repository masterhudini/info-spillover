# 🚀 Complete Setup Guide
## Hierarchical Sentiment Spillover Analysis Pipeline

This guide will walk you through setting up the complete hierarchical sentiment analysis pipeline for cryptocurrency information spillover detection.

---

## 📋 Prerequisites

- **Python 3.8+**
- **Google Cloud Account** with BigQuery enabled
- **Git** for version control
- **10GB+ free disk space** (for data and models)

---

## 🔧 Step-by-Step Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd info_spillover

# Create and activate virtual environment
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Google Cloud Configuration ⭐

This is **the most important step** - the pipeline requires BigQuery for data processing.

#### Option A: Service Account (Production/Recommended)

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Note down your **Project ID**

2. **Enable BigQuery API**
   - Go to APIs & Services → Library
   - Search for "BigQuery API"
   - Click **Enable**

3. **Create Service Account**
   - Go to IAM & Admin → Service Accounts
   - Click **Create Service Account**
   - Name: `hierarchical-sentiment-pipeline`
   - Grant roles:
     - `BigQuery Data Editor`
     - `BigQuery Job User`
     - `BigQuery User`

4. **Download Key File**
   - Click on your service account
   - Go to **Keys** tab
   - Click **Add Key** → **Create New Key** → **JSON**
   - Save the file securely (e.g., `~/gcp-keys/sentiment-pipeline-key.json`)

5. **Set Environment Variable**
   ```bash
   # Linux/Mac (add to ~/.bashrc or ~/.zshrc for persistence)
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

   # Windows (PowerShell)
   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\key.json"
   ```

#### Option B: User Authentication (Development)

```bash
# Install Google Cloud SDK
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate with your account
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 3. Verify Google Cloud Setup ✅

```bash
# Test your configuration
python scripts/test_gcp_setup.py

# Should output:
# ✅ Google Cloud Platform is properly configured
# ✅ BigQuery authentication successful
# ✅ Data pipeline is ready to run
```

If this fails, check the troubleshooting section below.

### 4. Configuration

Edit `experiments/configs/hierarchical_config.yaml` if needed:

```yaml
# Key settings you might want to adjust:
data:
  start_date: "2021-01-01"  # Analysis start date
  end_date: "2023-12-31"    # Analysis end date

hierarchical_model:
  subreddit_model_type: "lstm"  # or "transformer"
  max_epochs: 100               # Training epochs

backtesting:
  initial_capital: 100000       # Starting portfolio value
  transaction_costs: 0.001      # 0.1% per trade
```

### 5. Run the Pipeline! 🎉

```bash
# Execute the complete pipeline
python src/main_pipeline.py

# This will run all 5 steps:
# 1. 📊 Data processing and feature engineering
# 2. 🔄 Spillover analysis (Diebold-Yilmaz)
# 3. 🧠 Hierarchical modeling (LSTM + GNN)
# 4. 💼 Economic evaluation and backtesting
# 5. 📋 Report generation
```

**Expected runtime:** 30-60 minutes (depending on hardware and data size)

---

## 📊 Expected Output

After successful execution, you'll find results in `results/hierarchical_analysis/`:

```
results/hierarchical_analysis/
├── processed_data/
│   ├── hierarchical_features.parquet    # Engineered features
│   └── granger_causality_network.gml    # Network structure
├── spillover_analysis/
│   ├── spillover_heatmap.png           # Spillover visualization
│   ├── net_spillovers.png              # Net spillover effects
│   └── spillover_summary.json          # Quantitative results
├── models/
│   └── hierarchical_sentiment_model.*  # Trained ML models
├── backtesting/
│   ├── portfolio_performance.csv       # Trading results
│   ├── cumulative_returns.png          # Performance charts
│   └── performance_metrics.json        # Risk metrics
├── comprehensive_report.json           # Complete results
└── report.md                          # Human-readable report
```

## 📈 Key Results to Look For

- **Total Spillover Index**: Measure of interconnectedness (typically 20-60%)
- **Sharpe Ratio**: Risk-adjusted returns (>0.5 is good, >1.0 is excellent)
- **Annual Return**: Portfolio performance (compare to Bitcoin benchmark)
- **Max Drawdown**: Worst loss period (should be <20% for good strategies)

---

## 🐛 Troubleshooting

### Google Cloud Authentication Issues

**Problem**: `❌ Google Cloud authentication failed!`

**Solutions**:
1. **Check credentials file**:
   ```bash
   # Verify file exists and has correct permissions
   ls -la $GOOGLE_APPLICATION_CREDENTIALS

   # File should be readable and contain JSON
   cat $GOOGLE_APPLICATION_CREDENTIALS | head -5
   ```

2. **Check environment variable**:
   ```bash
   # Should output your key file path
   echo $GOOGLE_APPLICATION_CREDENTIALS
   ```

3. **Test gcloud CLI**:
   ```bash
   gcloud auth list
   gcloud config get-value project
   ```

### BigQuery Permission Issues

**Problem**: `❌ BigQuery access test failed!`

**Solutions**:
1. **Verify service account roles** in Google Cloud Console
2. **Enable BigQuery API** for your project
3. **Check project billing** (BigQuery requires billing enabled)

### Data Loading Issues

**Problem**: `No data returned from BigQuery`

This is expected on first run! The pipeline will:
1. Create BigQuery datasets automatically
2. Load data from `~/gcs/raw/` on first execution
3. Process and analyze the loaded data

Ensure your data is in the correct location:
```bash
ls ~/gcs/raw/posts_n_comments/  # Should show JSON files
ls ~/gcs/raw/prices/           # Should show CSV files
```

### Memory/Performance Issues

**Problem**: `OutOfMemoryError` or very slow processing

**Solutions**:
1. **Reduce date range** in config file
2. **Increase system RAM** or use cloud instance
3. **Enable GPU** training if available:
   ```yaml
   computing:
     use_gpu: true
   ```

---

## 🚀 Advanced Usage

### Individual Components

Run specific parts of the pipeline:

```bash
# Only data processing
python src/data/hierarchical_data_processor.py

# Only spillover analysis
python src/analysis/diebold_yilmaz_spillover.py

# Only backtesting
python src/evaluation/economic_evaluation.py
```

### MLFlow Tracking

Monitor experiments:

```bash
# Start MLFlow UI
mlflow ui --host 0.0.0.0 --port 5000

# Visit: http://localhost:5000
```

### Configuration Options

Key parameters to experiment with:

- **Model Architecture**: `lstm` vs `transformer`
- **GNN Type**: `GAT` vs `GCN` vs `GGNN`
- **Spillover Windows**: Adjust `window_size` and `step_size`
- **Trading Costs**: Modify `transaction_costs` for different markets

---

## ✅ Success Criteria

You know everything is working when:

1. **✅ GCP Test Passes**: `python scripts/test_gcp_setup.py`
2. **✅ Pipeline Completes**: No errors in main execution
3. **✅ Results Generated**: Files created in output directory
4. **✅ Reasonable Metrics**: Spillover index >10%, Sharpe ratio >0

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs carefully** - errors are usually descriptive
2. **Run individual components** to isolate problems
3. **Verify data availability** in `~/gcs/raw/`
4. **Test Google Cloud setup** with the test script
5. **Check system resources** (RAM, disk space)

---

## 📚 Next Steps

Once the pipeline is working:

1. **Analyze results** in the generated reports
2. **Experiment with parameters** in the config file
3. **Add new data sources** by extending the data processor
4. **Implement custom trading strategies** in the backtester
5. **Scale to real-time processing** using streaming data

---

*This setup guide covers the complete end-to-end configuration of the hierarchical sentiment spillover analysis pipeline. Follow each step carefully and you'll have a world-class financial AI system running!* 🎯