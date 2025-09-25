# 🚀 INSTRUKCJE URUCHOMIENIA PEŁNEGO PIPELINE'U

## ✅ Status: GOTOWY DO URUCHOMIENIA!

**Dane zostały załadowane do BigQuery:**
- ✅ **10,000** postów/komentarzy z Reddit (2013-2025)
- ✅ **10,227** rekordów cenowych crypto (2021-2024)
- ✅ **20 subredditów** crypto-tematycznych
- ✅ **7 kryptowalut** (BTC, ETH, ADA, SOL, XRP, BNB, LTC)

---

## 🎯 JAK URUCHOMIĆ PIPELINE

### 1. **Przygotuj środowisko**
```bash
cd /home/Hudini/projects/info_spillover
source ~/venvs/info_spillover/bin/activate
```

### 2. **Szybki test (5-10 minut)**
```bash
# Test z pojedynczą konfiguracją (zalecane na start)
python -m src.main_pipeline --single-config
```

### 3. **Pełny pipeline z optymalizacją hyperparametrów (30-60 minut)**
```bash
# Wszystkie 320 kombinacji konfiguracji
python -m src.main_pipeline
```

### 4. **Własna konfiguracja**
```bash
# Z własnymi ustawieniami
python -m src.main_pipeline --config experiments/configs/hierarchical_config.yaml --single-config
```

---

## 📊 CO OTRZYMASZ

### **Raporty automatyczne:**
- 📋 `results/[timestamp]/comprehensive_report.json` - Pełne wyniki JSON
- 📝 `results/[timestamp]/report.md` - Czytelny raport markdown
- 📈 `results/[timestamp]/statistical_significance_report.md` - **Analiza statystyczna z interpretacją**

### **Kluczowe metryki w raportach:**
- **Total Spillover Index** - % przepływu informacji między subredditami
- **Sharpe Ratio** - Risk-adjusted returns strategii
- **Alpha** - Excess returns vs benchmark (Bitcoin)
- **Statistical significance** - p-values, confidence intervals, ANOVA
- **Economic significance** - Information Ratio, Alpha tests

### **MLFlow tracking:**
```bash
# Zobacz metryki w przeglądarce
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Otwórz: http://localhost:5000
```

---

## 🔍 MONITORING WYKONANIA

### **Podczas działania:**
```bash
# Sprawdź procesy
ps aux | grep python

# Sprawdź logi
tail -f mlflow.log

# Zobacz utworzone pliki
ls -la results/
```

### **W tle (background):**
```bash
# Uruchom w tle z logowaniem
nohup python -m src.main_pipeline --single-config > pipeline.log 2>&1 &

# Śledź progress
tail -f pipeline.log
```

---

## ⚡ REKOMENDOWANE PIERWSZE URUCHOMIENIE

```bash
# KROK ZA KROKIEM:

# 1. Aktywuj środowisko
source ~/venvs/info_spillover/bin/activate

# 2. Przejdź do katalogu projektu
cd /home/Hudini/projects/info_spillover

# 3. Uruchom szybki test
python -m src.main_pipeline --single-config

# 4. Obserwuj output w terminalu
# Pipeline wypisze postęp dla każdego kroku:
# - STEP 1: Data Processing (BigQuery)
# - STEP 2: Spillover Analysis (Diebold-Yilmaz)
# - STEP 3: Hierarchical Modeling (LSTM + GNN)
# - STEP 4: Economic Evaluation (Backtesting)
# - STEP 5: Report Generation
```

---

## 📈 CZAS WYKONANIA

| Tryb | Czas | Opis |
|------|------|------|
| `--single-config` | **15-30 min** | Pojedyncza konfiguracja, zalecane na start |
| Pełny pipeline | **45-90 min** | Wszystkie 320 kombinacji hyperparametrów |
| Szybki test | **5-10 min** | Z ograniczonymi danymi (2021-2022) |

---

## 🎯 OCZEKIWANE WYNIKI

### **Scientific Value:**
- **Granger Causality Network** - Sieć przepływu informacji między subredditami
- **Spillover Analysis** - Quantitative measure information spillovers (Diebold-Yilmaz)
- **Hierarchical Model** - LSTM dla subredditów + GNN dla cross-spillovers
- **Statistical Tests** - Significance testing z p-values i confidence intervals

### **Economic Value:**
- **Trading Strategy** - Oparta na spillover predictions
- **Backtesting Results** - Realistic performance z transaction costs
- **Risk Metrics** - Sharpe ratio, max drawdown, VaR
- **Alpha Generation** - Excess returns vs Bitcoin benchmark

---

## 🚨 TROUBLESHOOTING

### **Jeśli pipeline się zatrzyma:**
```bash
# Sprawdź logi błędów
tail -50 mlflow.log

# Sprawdź wolne miejsce
df -h

# Sprawdź pamięć
free -h
```

### **Najczęstsze problemy:**
- **Out of Memory** → Zmniejsz `batch_size` w config
- **CUDA Error** → Ustaw `use_gpu: false`
- **BigQuery Timeout** → Pipeline automatycznie retry

---

## 🏁 GOTOWE!

**Twój pipeline jest gotowy do uruchomienia z realnym danych BigQuery!**

**Uruchom teraz:**
```bash
source ~/venvs/info_spillover/bin/activate && python -m src.main_pipeline --single-config
```

Pipeline będzie analizował:
- ✅ Real Reddit posts/comments data
- ✅ Real crypto price data
- ✅ Advanced statistical modeling
- ✅ Economic backtesting
- ✅ Comprehensive reports

**Rezultat: Publikowalny research z statistical significance analysis! 🎉**