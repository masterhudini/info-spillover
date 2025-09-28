# Claude Code - Projekt Info Spillover - Kontekst i Zasady

## 🎯 **Cel Główny Projektu**
Analiza spillover informacyjnego między subredditami cryptocurrency a cenami kryptowalut przy użyciu hierarchicznych modeli sentiment analysis z grafowymi sieciami neuronowymi (GNN).

Badanie oparte na metodologii Diebold-Yilmaz spillover analysis z nowoczesnymi technikami deep learning.

---

## 🏛️ **Podstawowe Zasady Pracy**

### ⚠️ **NAJWAŻNIEJSZA ZASADA**
> **"żadnych głupich hot fixów, tylko dokładnie, przeczytaj dokumentację, źródła naukowe itp."**

**Zawsze:**
1. **Czytaj dokumentację** - sprawdź wszystkie pliki README, konfiguracje, komentarze w kodzie
2. **Analizuj całościowo** - zrozum architekturę przed wprowadzeniem zmian
3. **Szukaj naukowego uzasadnienia** - każda implementacja powinna być oparta na solid theoretical foundation
4. **Testuj systematycznie** - nie rób zmian bez testowania
5. **Używaj właściwych narzędzi i wzorców** - nie wynajduj koła na nowo

**Nigdy nie:**
- Rób quick fixes bez zrozumienia problemu
- Wprowadzaj temporary patches
- Ignoruj błędów/warnings
- Zmieniaj konfiguracji bez uzasadnienia

---

## 🔧 **Środowisko Pracy**

### **Wirtualne Środowisko**
```bash
# ZAWSZE aktywuj przed pracą:
source ~/venvs/info_spillover/bin/activate

# Lokalizacja: /home/Hudini/venvs/info_spillover/
# NIE tworz nowych venv - używaj istniejącego
```

### **Testy i Pipeline**
```bash
# Szybki test pipeline:
source ~/venvs/info_spillover/bin/activate && timeout 120 python -m src.main_pipeline --config test_pipeline_quick.yaml --single-config

# Główny config: /home/Hudini/projects/info_spillover/experiments/configs/config.yaml
```

---

## 🗄️ **Dane i Źródła**

### **Hierarchia Ważności Źródeł Danych:**

1. **BigQuery (NAJWAŻNIEJSZE)** 🥇
   - Google Cloud BigQuery tables
   - `project.info_spillover.posts_comments`
   - `project.info_spillover.crypto_prices`
   - **Zawsze sprawdź BigQuery przed lokalnymi danymi**

2. **Lokalne Dane (backup/dev)** 🥈
   - `/home/Hudini/projects/info_spillover/data/`
   - CSV files, JSON exports
   - Używaj tylko gdy BigQuery niedostępne

3. **Syntetyczne Dane (tylko do testów)** 🥉
   - Generowane automatycznie dla dev/test
   - NIE używaj do produkcji

### **Dostęp do BigQuery:**
```python
from src.data.bigquery_client import BigQueryClient
client = BigQueryClient(project_id="your-project", dataset_id="info_spillover")
```

---

## 🏗️ **Architektura Modeli**

### **Model Hierarchiczny (Główny)**

```
Hierarchical Model Architecture:
├── Level 1: Subreddit-specific LSTMs
│   ├── SubredditLSTM dla każdego subreddita
│   ├── Input: sentiment features (24 dims typically)
│   └── Output: local predictions + embeddings
├── Level 2: Graph Neural Network
│   ├── SpilloverGNN (GAT/GCN layers)
│   ├── Input: node embeddings + spillover network
│   └── Output: cross-subreddit spillover effects
└── Fusion Layer: Final predictions
    ├── Combines L1 + L2 outputs
    └── Multi-task loss (MSE + CrossEntropy)
```

### **Kluczowe Komponenty:**
- `HierarchicalDataModule` - data loading & batching
- `HierarchicalBatchSampler` - multi-subreddit sampling
- `HierarchicalCollator` - custom batch construction
- `HierarchicalDataset` - PyTorch compatibility wrapper

---

## 📊 **Diebold-Yilmaz Spillover Analysis**

### **Metodologia:**
1. **VAR Model Fitting** - Vector Autoregression na sentiment data
2. **Variance Decomposition** - dekompozycja wariancji prediction errors
3. **Spillover Network Creation** - macierz spillover jako graf
4. **NetworkX Integration** - graf jako input do GNN

### **Kluczowe Pliki:**
- `src/analysis/diebold_yilmaz_spillover.py`
- Implementuje Diebold & Yilmaz (2012) methodology
- Input: multi-variate time series per subreddit
- Output: directed graph z spillover weights

---

## 🔄 **Pipeline Flow**

### **Główny Przepływ:**
```
1. Data Collection (BigQuery)
   ↓
2. Feature Engineering (sentiment, technical indicators)
   ↓
3. Hierarchical Data Processing (time series preparation)
   ↓
4. Diebold-Yilmaz Spillover Analysis (network creation)
   ↓
5. Hierarchical Model Training (LSTM + GNN)
   ↓
6. MLFlow Logging & Evaluation
```

### **Entry Points:**
- `src/main_pipeline.py` - główny pipeline
- `src/training_pipeline.py` - tylko training
- Konfig: `experiments/configs/config.yaml`

---

## 🧬 **Hierarchical Model - Technical Details**

### **Zrefactorowana Architektura (2024-12):**

#### **Data Flow:**
```
Raw Data → HierarchicalDataset → HierarchicalBatchSampler → HierarchicalCollator → Model
```

#### **Batch Structure:**
```python
# Batch format z HierarchicalCollator:
batch = (subreddit_data, graph_data, targets)

subreddit_data: Dict[str, torch.Tensor]  # {subreddit: sequences}
graph_data: GraphData object             # .subreddit_names, .edge_index, .edge_attr
targets: torch.Tensor                    # Concatenated targets (variable length)
```

#### **Robustność:**
- **Edge Validation**: sprawdza bounds dla graph indices
- **Fallback Mechanisms**: L1-only predictions gdy GNN fails
- **Dynamic Graph Creation**: tylko dla dostępnych subreddits
- **Tensor Shape Handling**: różne długości sequences per subreddit

---

## 🐛 **Common Issues & Solutions**

### **"not enough values to unpack"**
- ❌ Problem: batch unpacking errors
- ✅ Rozwiązanie: sprawdź HierarchicalCollator return format

### **"'Tensor' object has no attribute 'items'"**
- ❌ Problem: model expects dict, gets tensor
- ✅ Rozwiązanie: sprawdź subreddit_data structure

### **"input.size(-1) must be equal to input_size"**
- ❌ Problem: LSTM dimension mismatch
- ✅ Rozwiązanie: sprawdź input_dim w model initialization

### **"index X is out of bounds for dimension 0"**
- ❌ Problem: graph edge indices > num_nodes
- ✅ Rozwiązanie: edge validation + fallback graph creation

### **BigQuery Authentication Errors**
```bash
# Sprawdź credentials:
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

---

## 📦 **Dependencies & Tools**

### **Core Libraries:**
- `pytorch-lightning` - model training framework
- `torch-geometric` - GNN operations
- `networkx` - graph processing
- `google-cloud-bigquery` - data access
- `mlflow` - experiment tracking
- `dvc` - data versioning

### **Development:**
- `pytest` - testing framework
- `black` - code formatting
- `ruff` - linting

---

## 🔬 **Scientific Background**

### **Key Papers:**
1. **Diebold & Yilmaz (2012)** - "Better to give than to receive: Predictive directional measurement of volatility spillovers"
2. **Li et al. (2015)** - "Gated Graph Sequence Neural Networks"
3. **Veličković et al. (2017)** - "Graph Attention Networks"

### **Research Context:**
- **Information Spillover Theory**: how sentiment flows between markets
- **Behavioral Finance**: social media impact on trading
- **Network Effects**: interconnectedness in crypto markets

---

## 🚀 **MLFlow Integration**

### **Experiment Tracking:**
```python
# Auto-logging enabled w pipeline
mlflow.pytorch.autolog()

# Metrics tracked:
# - training_loss, val_loss
# - model parameters & gradients
# - spillover network statistics
# - prediction accuracy & correlation
```

### **Model Registry:**
- Models automatycznie rejestrowane po training
- Versioning based on performance metrics
- Integration z BigQuery dla model serving

---

## 📁 **Project Structure Overview**

```
info_spillover/
├── src/
│   ├── data/                     # Data processing
│   │   ├── bigquery_client.py   # BigQuery integration
│   │   └── hierarchical_data_processor.py
│   ├── models/                   # Model implementations
│   │   └── hierarchical_models.py  # Main hierarchical model
│   ├── analysis/                 # Analysis modules
│   │   └── diebold_yilmaz_spillover.py
│   └── main_pipeline.py         # Entry point
├── experiments/
│   └── configs/                 # Configuration files
├── data/                        # Local data (backup)
└── notebooks/                   # Jupyter analysis
```

---

## ✅ **Validation Checklist**

Przed każdą implementacją:

- [ ] Przeczytałem dokumentację związaną z obszarem
- [ ] Zrozumiałem scientific background
- [ ] Sprawdziłem existing patterns w kodzie
- [ ] Przetestowałem z venv activation
- [ ] Sprawdziłem BigQuery availability
- [ ] Uruchomiłem tests przed commit
- [ ] Code follows project conventions
- [ ] No temporary fixes lub hot patches

---

## 📝 **Development History Notes**

### **Major Refactoring (Dec 2024):**
- **Problem**: Batch unpacking failures w hierarchical model
- **Root Cause**: Architectural mismatch - model expected dict, DataLoader returned tensors
- **Solution**: Complete refactoring z HierarchicalBatchSampler + HierarchicalCollator + HierarchicalDataset
- **Result**: Robust multi-subreddit batching z proper error handling

### **Lessons Learned:**
1. **Architecture-first approach** beats quick fixes
2. **Graph validation** critical dla GNN stability
3. **Tensor shape consistency** requires careful collation
4. **Fallback mechanisms** essential dla robust production models

---

*Dokument zaktualizowany: December 2024*
*Następna aktualizacja: po kolejnych major changes*