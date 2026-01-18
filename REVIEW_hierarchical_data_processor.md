# PRZEGLĄD: hierarchical_data_processor.py
**Data:** 2026-01-11
**Status:** Znaleziono problemy wymagające poprawy

---

## ✅ CO JEST POPRAWNE

### 1. Walidacja sentiment scores (linie 78-120)
✅ Zakres VADER: [-1, +1] (poprawiony)
✅ Walidacja neutral scores: |score| > 0.3 (poprawne)
✅ Metryki jakości: mean, std, median, skewness, kurtosis

### 2. Fear/Greed/FOMO scores (linie 213-216)
✅ Fear: `sentiment_score < -0.5` (poprawne dla [-1, +1])
✅ Greed: `sentiment_score > 0.5` (poprawne)
✅ FOMO: `0.2 < sentiment_score < 0.5` (poprawne)

### 3. Load and validate data (linie 478-589)
✅ Używa `get_post_sentiment_aggregation()` - pobiera 5-min dane
✅ Walidacja jakości danych
✅ Strict checks: min_records=100, min_subreddits=3, min_days=5

### 4. Fallback query (linie 515-532)
✅ Używa `posts_comments_enhanced_sentiment` (ale **NIE UŻYWANY**, bo hasattr sprawdza get_post_sentiment_aggregation)
✅ Kolumny: new_sentiment_label, new_sentiment_compound

---

## ❌ PROBLEMY DO NAPRAWIENIA

### PROBLEM 1: Niepoprawne tworzenie compound_sentiment (linie 491-495)
**Kod:**
```python
df['compound_sentiment'] = (
    df['avg_positive_sentiment'].fillna(0) -
    df['avg_negative_sentiment'].fillna(0)
)
```

**Problem:**
- BigQuery już zwraca `avg_compound_sentiment` z VADER!
- Ta kalkulacja (positive - negative) **NIE JEST** VADER compound!
- Nadpisujesz prawdziwy VADER compound niepoprawnym obliczeniem

**Rozwiązanie:**
```python
# Użyj prawdziwego VADER compound z BigQuery
if 'avg_compound_sentiment' in df.columns:
    df['compound_sentiment'] = df['avg_compound_sentiment']
else:
    # Fallback tylko jeśli nie ma avg_compound_sentiment
    df['compound_sentiment'] = (
        df['avg_positive_sentiment'].fillna(0) -
        df['avg_negative_sentiment'].fillna(0)
    )
```

---

### PROBLEM 2: Niepoprawna kalkulacja compound_sentiment w feature engineer (linie 166-183)
**Kod:**
```python
def compute_compound_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment_label'].map(label_map)

    # To NIE JEST VADER compound!
    df['compound_sentiment'] = df['sentiment_numeric'] * df['sentiment_score']
    df['compound_sentiment'] = np.clip(df['compound_sentiment'], -1, 1)
```

**Problem:**
- `sentiment_numeric * sentiment_score` to NIE JEST VADER compound
- Dla agregowanych danych (5-min), `compound_sentiment` **już istnieje** z load_and_validate_data
- Ta metoda nadpisuje prawdziwy compound niepoprawnym

**Rozwiązanie:**
```python
def compute_compound_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
    """Use existing compound_sentiment or create from label mapping"""

    # If compound_sentiment already exists (from aggregated data), use it
    if 'compound_sentiment' in df.columns:
        logger.info("Using existing compound_sentiment from aggregated data")
        return df

    # Otherwise, create from label mapping (for raw data only)
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_numeric'] = df['sentiment_label'].map(label_map)

    # Simple mapping for raw data without VADER scores
    df['compound_sentiment'] = df['sentiment_numeric'] * abs(df['sentiment_score'])
    df['compound_sentiment'] = np.clip(df['compound_sentiment'], -1, 1)

    return df
```

---

### PROBLEM 3: Temporal aggregates zakłada dane dzienne (linie 220-279)
**Kod:**
```python
def compute_temporal_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
    # For each time window in DAYS (data is daily aggregated)
    for window_days in self.window_hours:  # Note: misnamed variable, actually contains days
        window_periods = max(1, window_days)
```

**Problem:**
- Komentarz mówi "data is daily aggregated"
- Ale teraz mamy **5-minutowe agregaty**!
- `window_hours = [1, 6, 24]` prawdopodobnie to GODZINY, nie dni
- Dla 5-min danych: 1h = 12 binów, 6h = 72 bins, 24h = 288 bins

**Rozwiązanie:**
```python
def compute_temporal_aggregates(self, df: pd.DataFrame,
                                bin_minutes: int = 5) -> pd.DataFrame:
    """
    Compute temporal aggregates with time windows

    Args:
        df: DataFrame with 5-minute aggregated data
        bin_minutes: Minutes per bin (default 5 for 5-minute aggregation)
    """
    logger.info("Computing temporal aggregates...")

    # Ensure proper datetime index
    df = df.set_index('post_created_utc').sort_index()

    # Initialize results container
    aggregated_features = []

    for subreddit in df['subreddit'].unique():
        logger.info(f"Processing temporal aggregates for {subreddit}")

        subreddit_data = df[df['subreddit'] == subreddit].copy()

        # For each time window (in hours)
        for window_hours in self.window_hours:  # e.g., [1, 6, 24]
            # Convert hours to number of 5-minute bins
            window_periods = int(window_hours * 60 / bin_minutes)  # e.g., 1h = 12 bins

            # Rolling statistics over 5-minute bins
            rolling = subreddit_data['compound_sentiment'].rolling(
                window_periods, min_periods=1
            )

            subreddit_data[f'sentiment_mean_{window_hours}h'] = rolling.mean()
            subreddit_data[f'sentiment_std_{window_hours}h'] = rolling.std()
            # ... rest of rolling stats
```

---

### PROBLEM 4: Granger causality resample freq (linie 292-325)
**Kod:**
```python
def prepare_time_series(self, df: pd.DataFrame,
                      freq: str = '1H') -> Dict[str, pd.Series]:
    # Resample to regular frequency, forward-fill missing values
    ts = subreddit_data['compound_sentiment'].resample(freq).mean()
```

**Problem:**
- Domyślnie resample do '1H' (godziny)
- Ale mamy już 5-minutowe agregaty!
- Resampling może być niepotrzebny lub powinien używać '5T' (5 minutes)

**Rozwiązanie:**
```python
def prepare_time_series(self, df: pd.DataFrame,
                      freq: str = '5T') -> Dict[str, pd.Series]:
    """
    Prepare time series for Granger causality testing

    Args:
        freq: Resampling frequency ('5T' = 5 minutes, '1H' = 1 hour)
    """
    logger.info(f"Preparing time series for Granger causality (freq={freq})...")

    # For 5-minute data, use '5T' to preserve granularity
    # For hourly analysis, use '1H' to reduce noise
```

---

### PROBLEM 5: Timestamp column inconsistency (linie 302-311)
**Kod:**
```python
timestamp_col = 'created_utc' if 'created_utc' in subreddit_data.columns else 'post_created_utc'
```

**Problem:**
- Kod sprawdza czy istnieje `created_utc` lub `post_created_utc`
- Ale `get_post_sentiment_aggregation()` ZAWSZE zwraca `post_created_utc`
- Ten check jest zbędny dla agregowanych danych

**Rozwiązanie:**
```python
# For aggregated data, always use 'post_created_utc'
timestamp_col = 'post_created_utc'

if timestamp_col not in subreddit_data.columns:
    logger.warning(f"Missing {timestamp_col} column for {subreddit}, skipping")
    continue
```

---

### PROBLEM 6: Fillna method deprecated (linia 315)
**Kod:**
```python
ts = ts.fillna(method='ffill', limit=3)  # Max 3 consecutive fills
```

**Problem:**
- Pandas deprecated `method='ffill'` parameter
- Nowa składnia: `ts.ffill(limit=3)`

**Rozwiązanie:**
```python
ts = ts.ffill(limit=3)  # Forward-fill max 3 consecutive NaNs
```

---

## 📊 PODSUMOWANIE PRIORYTETÓW

### KRYTYCZNE (muszą być naprawione):
1. ❌ **PROBLEM 1**: Niepoprawne tworzenie compound_sentiment (nadpisuje VADER)
2. ❌ **PROBLEM 2**: Feature engineer nadpisuje compound_sentiment

### WYSOKIE (powinny być naprawione):
3. ⚠️ **PROBLEM 3**: Temporal aggregates zakłada dni zamiast 5-min bins
4. ⚠️ **PROBLEM 4**: Granger causality resample freq (1H vs 5T)

### NISKIE (kosmetyczne):
5. 🔧 **PROBLEM 5**: Timestamp column inconsistency
6. 🔧 **PROBLEM 6**: Deprecated fillna method

---

## ✅ PLAN NAPRAWY

### Krok 1: Napraw compound_sentiment (KRYTYCZNE)
**Plik:** hierarchical_data_processor.py
**Linie:** 491-495, 166-183

**Zmiana:**
```python
# W load_and_validate_data (linie 491-495):
if 'avg_compound_sentiment' in df.columns:
    df['compound_sentiment'] = df['avg_compound_sentiment']
else:
    # Fallback calculation only if VADER compound not available
    df['compound_sentiment'] = (
        df['avg_positive_sentiment'].fillna(0) -
        df['avg_negative_sentiment'].fillna(0)
    )

# W HierarchicalFeatureEngineer.compute_compound_sentiment (linie 166-183):
def compute_compound_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
    if 'compound_sentiment' in df.columns:
        logger.info("Using existing VADER compound_sentiment")

        # Create sentiment_numeric for compatibility
        if 'sentiment_label' in df.columns:
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            df['sentiment_numeric'] = df['sentiment_label'].map(label_map)

        return df

    # Original logic only if compound_sentiment doesn't exist...
```

### Krok 2: Napraw temporal aggregates (WYSOKIE)
**Plik:** hierarchical_data_processor.py
**Linie:** 220-279

**Zmiana:**
- Dodaj parametr `bin_minutes=5`
- Konwertuj `window_hours` na liczbę binów: `window_periods = window_hours * 60 / bin_minutes`
- Zmień nazwy kolumn z `_Xd` na `_Xh` (hours zamiast days)

### Krok 3: Napraw Granger causality freq (WYSOKIE)
**Plik:** hierarchical_data_processor.py
**Linie:** 292-325

**Zmiana:**
- Zmień domyślny freq z '1H' na '5T'
- Dodaj dokumentację o wpływie freq na wyniki

### Krok 4: Napraw deprecated pandas (NISKIE)
**Plik:** hierarchical_data_processor.py
**Linia:** 315

**Zmiana:**
```python
ts = ts.ffill(limit=3)  # Nowa składnia pandas
```

---

## 🧪 TESTY PO NAPRAWIE

Po wprowadzeniu poprawek, uruchom:

```bash
cd /home/Hudini/projects/info_spillover_enhanced
python3 test_data_processor.py
```

**Sprawdź:**
1. ✅ Compound sentiment jest z BigQuery (avg_compound_sentiment)
2. ✅ Zakres sentiment_score: [-0.94, 0.99] (VADER compound)
3. ✅ Temporal windows używają 5-min bins, nie dni
4. ✅ Granger causality używa 5T freq dla 5-min danych

---

**Pytanie do Ciebie:**
Czy chcesz żebym teraz wprowadził te poprawki, czy wolisz najpierw przejrzeć ten raport?
