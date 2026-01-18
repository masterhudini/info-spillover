# Audyt Transformacji - hierarchical_data_processor.py

## Podsumowanie krytycznych problemów

| # | Linia | Problem | Severity |
|---|-------|---------|----------|
| 1 | 921-928 | **merge_asof bez `by='symbol'`** - posty dostają losową cenę | CRITICAL |
| 2 | 181 | `compute_compound_sentiment` tylko castuje typ, nazwa myląca | LOW |
| 3 | 204-206 | `log1p` poprawne, ale brak walidacji | MEDIUM |
| 4 | 907 | `dropna(subset=['price'])` - ciche usuwanie, brak alertu | MEDIUM |
| 5 | 350-360 | `resample().mean()` + `ffill()` + `diff()` + `dropna()` - kaskada bez walidacji | HIGH |
| 6 | 906-907 | `resample().last()` + `dropna` - brak sprawdzenia coverage | MEDIUM |

---

## Szczegółowy audyt linia po linii

### Class: DataQualityValidator

#### L46-47: `pd.to_datetime` conversion
```python
if df[timestamp_col].dtype == 'object':
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
```
**Status:** OK, ale brak walidacji
**Walidacja do dodania:**
- Sprawdzić ile wartości nie udało się przekonwertować (NaT)
- `validate_astype()` po konwersji

#### L56: `diff()` na timestamps
```python
time_diffs = df[timestamp_col].diff()
```
**Status:** OK
**Walidacja:** Pierwszy element będzie NaT - expected

---

### Class: HierarchicalFeatureEngineer

#### L181: `astype(float)` w compute_compound_sentiment
```python
df['compound_sentiment'] = df['compound_sentiment'].astype(float)
```
**Status:** MISLEADING NAME - funkcja nazywa się "compute" ale nic nie oblicza
**Walidacja do dodania:**
```python
validator.validate_astype(before, after, 'compound_sentiment', 'float')
validator.validate_sentiment_score(df['compound_sentiment'])
```

#### L192: `set_index` + `sort_index`
```python
df = df.set_index('post_created_utc').sort_index()
```
**Status:** OK
**Walidacja:** Sprawdzić czy index jest unique lub czy duplikaty są expected

#### L204-206: `log1p` na activity
```python
subreddit_data['activity_log'] = np.log1p(
    subreddit_data['post_count'] + subreddit_data['comment_count']
)
```
**Status:** POPRAWNE użycie log1p (obsługuje 0)
**Walidacja do dodania:**
```python
original_sum = subreddit_data['post_count'] + subreddit_data['comment_count']
validator.validate_log_transform(
    subreddit_data['activity_log'],
    'activity_log',
    original_min=original_sum.min()
)
```
**Uwaga:** Nazwa `activity_log` może być myląca (logarytm vs log aktywności)

#### L207: `diff()` na activity_log
```python
subreddit_data['activity_change'] = subreddit_data['activity_log'].diff()
```
**Status:** OK
**Walidacja do dodania:**
```python
validator.validate_diff(
    subreddit_data['activity_log'],
    subreddit_data['activity_change'],
    'activity_change',
    expected_nan_count=1  # per group!
)
```
**UWAGA:** To jest per-subreddit, więc każda grupa ma swój pierwszy NaN

#### L208: `diff()` na compound_sentiment
```python
subreddit_data['compound_change'] = subreddit_data['compound_sentiment'].diff()
```
**Status:** OK
**Walidacja:** j.w.

#### L209-211: `rolling().std()`
```python
rolling_std = subreddit_data['compound_sentiment'].rolling(
    window_bins_short, min_periods=1
).std()
```
**Status:** OK, min_periods=1 pozwala na wartości od początku
**Walidacja do dodania:**
```python
validator.validate_rolling(
    subreddit_data['compound_sentiment'],
    rolling_std,
    'compound_sentiment_std',
    window=window_bins_short,
    min_periods=1
)
```

#### L212: Division with replace(0, np.nan)
```python
subreddit_data['compound_shock'] = subreddit_data['compound_change'] / rolling_std.replace(0, np.nan)
```
**Status:** OK - poprawnie obsługuje dzielenie przez 0
**Walidacja do dodania:**
```python
validator.validate_range(subreddit_data['compound_shock'], 'compound_shock')
# Sprawdzić czy nie ma inf
assert not np.isinf(subreddit_data['compound_shock']).any()
```

#### L221-228: Rolling statistics
```python
rolling = subreddit_data['compound_sentiment'].rolling(window_bins, min_periods=1)
subreddit_data[f'sentiment_mean_{window_bins}bins'] = rolling.mean()
subreddit_data[f'sentiment_std_{window_bins}bins'] = rolling.std()
subreddit_data[f'sentiment_skew_{window_bins}bins'] = rolling.skew()
subreddit_data[f'sentiment_kurt_{window_bins}bins'] = rolling.apply(...)
```
**Status:** OK
**Walidacja:**
- `sentiment_mean` powinien być w [-1, 1]
- `sentiment_std` powinien być >= 0
- `sentiment_skew` i `sentiment_kurt` mogą być dowolne

#### L235-237: Volume-weighted sentiment
```python
subreddit_data[f'vwsentiment_{window_bins}bins'] = (
    (sentiment_vol.mean() * weights.mean()) / weights.mean()
).fillna(0)
```
**Status:** PODEJRZANE - `(x * y) / y = x` - to jest po prostu `sentiment_vol.mean()`!
**Problem:** Formuła jest matematycznie równoważna `rolling.mean()`, więc volume weighting nie działa
**Poprawna formuła powinna być:**
```python
# Weighted average: sum(sentiment * weight) / sum(weight)
(subreddit_data['compound_sentiment'] * subreddit_data['score']).rolling(...).sum() /
subreddit_data['score'].rolling(...).sum()
```

#### L240-244: Post count rolling
```python
subreddit_data[f'post_count_{window_bins}bins'] = (
    subreddit_data['compound_sentiment']
    .rolling(window_bins, min_periods=1)
    .count()
)
```
**Status:** MISLEADING - to liczy nie-null wartości sentiment, nie posty
**Walidacja:** Nazwa sugeruje liczbę postów, ale to count() na sentiment

#### L249: `pd.concat` aggregated_features
```python
result_df = pd.concat(aggregated_features, ignore_index=False)
```
**Status:** OK
**Walidacja do dodania:**
```python
# Suma długości części powinna równać się całości
expected_len = sum(len(part) for part in aggregated_features)
validator.validate_merge_row_count(
    pd.DataFrame({'_': range(expected_len)}),
    result_df,
    operation="concat_subreddits"
)
```

---

### Class: DieboldYilmazSpilloverBuilder

#### L350-352: `pivot_table` + `resample`
```python
pivot = (
    df.pivot_table(...)
    .sort_index()
)
pivot = pivot.resample(freq).mean()
```
**Status:** OK, ale może tworzyć NaN dla okresów bez danych
**Walidacja:**
```python
validator.validate_resample(df, pivot, freq, 'post_created_utc')
```

#### L351-352: `ffill(limit=...)`
```python
if self.ffill_limit > 0:
    pivot = pivot.ffill(limit=self.ffill_limit)
```
**Status:** OK
**Walidacja:** Sprawdzić ile NaN zostało po ffill

#### L358: `diff()` na wybranych subredditach
```python
pivot[self.diff_subreddits] = pivot[self.diff_subreddits].diff()
```
**Status:** OK
**Walidacja:** Pierwszy wiersz będzie NaN

#### L360: `dropna()` po diff
```python
pivot = pivot.dropna()
```
**Status:** POTENCJALNY PROBLEM
**Walidacja do dodania:**
```python
validator.validate_dropna(pivot_before, pivot, subset=list(pivot.columns), max_drop_pct=0.2)
```
**Uwaga:** Jeśli dużo subredditów ma różne zakresy czasowe, to może usunąć dużo danych

#### L386-390: FEVD normalization
```python
row_sums = decomposition.sum(axis=1, keepdims=True)
if np.any(row_sums == 0):
    raise ValueError("FEVD row sums contain zeros, cannot normalize")
normalized = decomposition / row_sums
```
**Status:** OK - poprawna walidacja przed dzieleniem

---

### Class: HierarchicalDataProcessor

#### L604: Column rename
```python
df = df.rename(columns={'date': 'post_created_utc'})
```
**Status:** OK

#### L608: `astype(float)`
```python
df['compound_sentiment'] = df['avg_compound_sentiment'].astype(float)
```
**Status:** OK
**Walidacja do dodania:**
```python
validator.validate_astype(df['avg_compound_sentiment'], df['compound_sentiment'], 'compound_sentiment', 'float')
```

#### L630: `drop_duplicates()`
```python
df = df.drop_duplicates()
```
**Status:** OK
**Walidacja:** Logować ile usunięto (już robione w linii 631)

---

### **CRITICAL: create_targets() method (L888-946)**

#### L897: datetime conversion
```python
df['post_created_utc'] = pd.to_datetime(df['post_created_utc']).dt.tz_localize(None).astype('datetime64[ns]')
```
**Status:** OK

#### L902-912: Price data resampling
```python
price_data = price_data.sort_values(['symbol', 'snapped_at']).copy()
resampled = []
for symbol, sdf in price_data.groupby('symbol'):
    sdf = sdf.set_index('snapped_at').sort_index()
    sdf = sdf.resample(self.price_resample_freq).last()
    sdf = sdf.dropna(subset=['price'])  # <-- L907
    sdf['symbol'] = symbol
    resampled.append(sdf.reset_index())
```

**L906: `resample().last()`**
**Status:** OK - bierze ostatnią cenę z okresu
**Problem:** Tworzy NaN dla okresów bez danych

**L907: `dropna(subset=['price'])`** ⚠️
**Status:** POTENTIAL ISSUE
**Problem:** Ciche usuwanie okresów bez cen. Dla krypto (24/7) nie powinno być gaps.
**Walidacja do dodania:**
```python
before_dropna = len(sdf)
sdf = sdf.dropna(subset=['price'])
after_dropna = len(sdf)
if before_dropna != after_dropna:
    logger.warning(f"Symbol {symbol}: dropped {before_dropna - after_dropna} periods without price")
    # Zbadać dlaczego są gaps
```

#### L914-916: Log returns calculation
```python
price_data['price_log_return'] = price_data.groupby('symbol')['price'].transform(
    lambda s: np.log(s).diff().shift(-1)
)
```
**Status:** OK matematycznie
- `np.log(s)` - logarytm ceny
- `.diff()` - różnica = log return
- `.shift(-1)` - przesunięcie w przód = **przyszły** return

**Walidacja do dodania:**
```python
validator.validate_groupby_transform(price_data, price_data['price_log_return'], 'symbol', 'log_return')
# Sprawdzić rozkład - log returns powinny być ~normal, mean ~0
```

#### L921-928: **CRITICAL BUG** - merge_asof ⚠️⚠️⚠️
```python
df = pd.merge_asof(
    df.sort_values('post_created_utc'),
    price_data.sort_values('snapped_at'),
    left_on='post_created_utc',
    right_on='snapped_at',
    direction='nearest',
    tolerance=pd.Timedelta(self.price_merge_tolerance)
)
```

**Status:** KRYTYCZNY BUG
**Problem:** Brak `by='symbol'` w merge_asof!

**Co się dzieje:**
1. `df` (Reddit posts) ma kolumny: `subreddit`, `post_created_utc`, ... (NIE ma `symbol`)
2. `price_data` ma kolumny: `symbol`, `snapped_at`, `price`, `price_log_return`
3. Merge łączy po **czasie**, ignorując symbol
4. Każdy post z r/cardano może dostać cenę BTC, ETH, ADA, lub SOL - zależnie od tego która jest najbliższa w czasie!

**Poprawka:**
1. Dodać mapowanie `subreddit → symbol` w konfiguracji
2. Dodać kolumnę `symbol` do `df` przed merge
3. Użyć `by='symbol'` w merge_asof

**Walidacja do dodania:**
```python
# PRZED MERGE:
validator.validate_merge_by_column(df, 'symbol', 'symbol', 'merge_asof_price')

# PO MERGE:
validator.validate_merge_row_count(df_before, df, 'merge_asof_price')
validator.validate_merge_asof_coverage(df_before, df, 'snapped_at', 'price', max_unmatched_pct=0.05)
```

#### L930: Sort after merge
```python
df = df.sort_values(['symbol', 'post_created_utc'])
```
**Status:** PROBLEM - `symbol` pochodzi z price_data, nie z mapowania subreddit
**Problem:** Sortowanie po `symbol` który został losowo przypisany

#### L931: Copy price_log_return to next_return
```python
df['next_return'] = df['price_log_return']
```
**Status:** OK technicznie, ale dane są błędne z powodu merge bug

#### L932-935: Return direction
```python
df['return_direction'] = np.where(
    df['next_return'] > self.return_direction_threshold, 1,
    np.where(df['next_return'] < -self.return_direction_threshold, -1, 0)
)
```
**Status:** OK technicznie
**Walidacja:**
```python
validator.validate_range(df['return_direction'], 'return_direction', min_val=-1, max_val=1)
```

#### L943: Sentiment shift
```python
df['next_sentiment'] = df.groupby('subreddit')['compound_sentiment'].shift(-1)
```
**Status:** OK
**Walidacja:**
```python
validator.validate_shift(df['compound_sentiment'], df['next_sentiment'], 'next_sentiment', -1)
```

#### L944: Sentiment change
```python
df['sentiment_change'] = df['next_sentiment'] - df['compound_sentiment']
```
**Status:** OK
**Walidacja:**
```python
# Zakres powinien być [-2, 2] bo sentiment jest w [-1, 1]
validator.validate_range(df['sentiment_change'], 'sentiment_change', min_val=-2, max_val=2)
```

---

### Pipeline validations (process_full_pipeline_chunked)

#### L851: dropna
```python
df = df.dropna(subset=['compound_sentiment', 'post_created_utc'])
```
**Status:** OK
**Walidacja do dodania:**
```python
validator.validate_dropna(df_before, df, ['compound_sentiment', 'post_created_utc'], max_drop_pct=0.1)
```

#### L854: Date filtering
```python
mask = df['post_created_utc'].dt.date.between(chunk_start, chunk_end)
df = df.loc[mask].copy()
```
**Status:** OK (usuwa overlap dane)
**Walidacja:** Logować ile usunięto przez filtr dat

#### L860: Column filtering
```python
df = df[keep_columns]
```
**Status:** OK, ale...
**UWAGA:** Jeśli `next_return` nie jest w `keep_columns`, zostanie usunięta, a potem walidacja w L873 fail!

#### L868: Final concat
```python
df_final = pd.concat(chunks, ignore_index=True)
```
**Status:** OK

---

## Priorytety napraw

### P0 - Critical (blokujące)
1. **L921-928**: Naprawić merge_asof - dodać mapowanie subreddit→symbol i `by='symbol'`

### P1 - High (poważne błędy logiczne)
2. **L235-237**: Naprawić volume-weighted sentiment - aktualna formuła nie waży
3. **L350-360**: Dodać walidację po kaskadzie resample+ffill+diff+dropna

### P2 - Medium (brakujące walidacje)
4. **L907**: Dodać alert gdy dropna usuwa ceny
5. **L204-212**: Dodać walidacje dla log1p, diff, rolling
6. **L914-916**: Walidacja log returns

### P3 - Low (kosmetyczne)
7. **L181**: Rename `compute_compound_sentiment` → `cast_compound_sentiment` lub dodać faktyczne obliczenia
8. **L240-244**: Rename `post_count_*` → `sentiment_count_*` lub naprawić logikę

---

## Rekomendowane zmiany kodu

### 1. Dodać mapowanie subreddit → symbol

W `.env`:
```
SUBREDDIT_SYMBOL_MAP={"Bitcoin":"BTC","ethereum":"ETH","ethtrader":"ETH","cardano":"ADA","solana":"SOL","CryptoCurrency":"BTC","CryptoMarkets":"BTC","BitcoinBeginners":"BTC","defi":"ETH","cryptomoonshots":"BTC"}
```

W kodzie przed merge:
```python
# Load mapping
symbol_map = json.loads(os.getenv("SUBREDDIT_SYMBOL_MAP", "{}"))

# Add symbol column to df based on subreddit
df['symbol'] = df['subreddit'].map(symbol_map)
if df['symbol'].isna().any():
    unmapped = df[df['symbol'].isna()]['subreddit'].unique()
    raise ValueError(f"Unmapped subreddits: {unmapped}")

# Now merge with by='symbol'
df = pd.merge_asof(
    df.sort_values(['symbol', 'post_created_utc']),
    price_data.sort_values(['symbol', 'snapped_at']),
    left_on='post_created_utc',
    right_on='snapped_at',
    by='symbol',
    direction='nearest',
    tolerance=pd.Timedelta(self.price_merge_tolerance)
)
```

### 2. Naprawić volume-weighted sentiment

```python
# Current (broken):
subreddit_data[f'vwsentiment_{window_bins}bins'] = (
    (sentiment_vol.mean() * weights.mean()) / weights.mean()
).fillna(0)

# Fixed:
weighted_sum = (subreddit_data['compound_sentiment'] * subreddit_data['score']).rolling(window_bins, min_periods=1).sum()
weight_sum = subreddit_data['score'].rolling(window_bins, min_periods=1).sum()
subreddit_data[f'vwsentiment_{window_bins}bins'] = (weighted_sum / weight_sum).fillna(0)
```

### 3. Dodać walidację po dropna na cenach

```python
# Before:
sdf = sdf.dropna(subset=['price'])

# After:
before_count = len(sdf)
sdf = sdf.dropna(subset=['price'])
dropped = before_count - len(sdf)
if dropped > 0:
    drop_pct = dropped / before_count
    logger.warning(f"Symbol {symbol}: dropped {dropped} ({drop_pct:.1%}) periods without price data")
    if drop_pct > 0.1:
        raise ValueError(f"Symbol {symbol}: too many missing prices ({drop_pct:.1%})")
```
