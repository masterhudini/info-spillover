# 🚀 Instrukcje uruchomienia Pipeline'u

## Problem: BigQuery tabele są puste
Wszystkie tabele w BigQuery są puste, więc musimy użyć lokalnych danych z GCS mount.

## 🔧 Przygotowanie środowiska

### 1. Aktywuj virtual environment
```bash
source ~/venvs/info_spillover/bin/activate
```

### 2. Sprawdź czy wszystkie dependencies są zainstalowane
```bash
pip install torch torch-geometric pytorch-lightning transformers yfinance db-dtypes google-cloud-bigquery pandas numpy scikit-learn scipy statsmodels networkx matplotlib seaborn mlflow
```

## 📊 Opcje uruchomienia

### OPCJA A: Szybki test na lokalnych danych JSON (POLECANA)

1. **Stwórz prostą konfigurację test_local.yaml:**
```yaml
experiment:
  name: "test_local_data"
  description: "Test with local JSON data"

data:
  # Użyj lokalnych danych zamiast BigQuery
  use_local_data: true
  local_data_path: "~/gcs/raw/posts_n_comments"
  start_date: "2021-01-01"
  end_date: "2021-12-31"

hierarchical_model:
  hidden_dim: 32
  max_epochs: 3  # Bardzo szybko
  batch_size: 16

mlflow:
  tracking_uri: "sqlite:///test_mlflow.db"
```

2. **Zmodyfikuj main_pipeline.py żeby obsługiwał lokalne dane:**
```python
# W step_1_data_processing() dodaj:
if self.config['data'].get('use_local_data', False):
    from src.data.local_data_processor import LocalDataProcessor
    local_processor = LocalDataProcessor(self.config['data']['local_data_path'])
    processed_data, network, processing_log = local_processor.process_full_pipeline(
        start_date=self.config['data']['start_date'],
        end_date=self.config['data']['end_date']
    )
else:
    # Istniejący kod BigQuery...
```

### OPCJA B: Pełny test z BigQuery (po naprawieniu danych)

1. **Najpierw załaduj dane do BigQuery:**
```bash
# Upload JSON files to BigQuery
python scripts/upload_json_to_bigquery.py --source ~/gcs/raw/posts_n_comments --dataset spillover_statistical_test
```

2. **Uruchom z właściwą konfiguracją:**
```bash
python -m src.main_pipeline --config experiments/configs/hierarchical_config.yaml --single-config
```

## 🎯 Proste uruchomienie pipeline'u

### Szybki test (5-10 minut):
```bash
cd /home/Hudini/projects/info_spillover
source ~/venvs/info_spillover/bin/activate

# Test na małych danych
python -m src.main_pipeline --config test_local.yaml --single-config
```

### Pełny pipeline (30-60 minut):
```bash
cd /home/Hudini/projects/info_spillover
source ~/venvs/info_spillover/bin/activate

# Wszystkie 320 kombinacji hyperparametrów
python -m src.main_pipeline

# Lub pojedyncza konfiguracja
python -m src.main_pipeline --single-config
```

## 📋 Monitoring wykonania

### Sprawdzanie statusu:
```bash
# Jeśli uruchomione w tle, sprawdzaj proces:
ps aux | grep python

# Sprawdź logi MLFlow:
tail -f mlflow.log

# Wyniki będą w:
ls -la results/
```

### Monitoring w czasie rzeczywistym:
```bash
# Uruchom w osobnym terminalu
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Otwórz http://localhost:5000 w przeglądarce
```

## 📊 Co otrzymasz na wyjściu

### Pliki wynikowe:
- `results/[timestamp]/comprehensive_report.json` - Pełny raport
- `results/[timestamp]/report.md` - Raport czytelny dla człowieka
- `results/[timestamp]/statistical_significance_report.md` - Analiza statystyczna
- MLFlow tracking - Metryki i modele

### Kluczowe metryki:
- **Spillover Index**: % przepływu informacji między subredditami
- **Sharpe Ratio**: Risk-adjusted returns strategii
- **Alpha**: Excess returns vs benchmark
- **Statistical significance**: p-values dla wszystkich testów

## 🔧 Troubleshooting

### Błędy dependencies:
```bash
pip install --upgrade [missing_package]
```

### Błędy pamięci:
```bash
# Zmniejsz batch_size i hidden_dim w config
```

### Błędy GPU:
```bash
# Wyłącz GPU w konfiguracji:
# use_gpu: false
```

## 🚀 Rekomendowane uruchomienie

**Dla pierwszego testu:**
```bash
cd /home/Hudini/projects/info_spillover
source ~/venvs/info_spillover/bin/activate

# Szybki test lokalny (po implementacji local_data_processor)
python quick_local_test.py

# Lub pełny pipeline z pojedynczą konfiguracją
python -m src.main_pipeline --single-config
```

Pipeline zajmie 15-45 minut w zależności od ilości danych i będzie generował szczegółowe raporty z analizą statystyczną spillover effects w krypto-social media.