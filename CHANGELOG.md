# Changelog - Data Processing Update (2026-01-11)

## Summary
Migrated from simple `prepare_data.py` to full `HierarchicalDataProcessor` pipeline with corrected BigQuery table references, VADER sentiment score ranges, and **5-minute temporal aggregation**.

## Changes Made

### 1. Environment Configuration (.env)
**Updated:**
- Fixed `.env` format (removed Python type annotations)
- Changed date range from 2024-2025 to **2021-02-01 to 2025-10-07**
- Added `BIGQUERY_SENTIMENT_TABLE="posts_comments_enhanced_sentiment"` for explicit table reference
- Verified against actual BigQuery data availability:
  - Sentiment data: 2017-01-05 to 2025-10-07 (653K rows)
  - Price data: 2018-01-01 to 2026-01-06 (6.2M rows, 5-minute granularity)

**Rationale:**
- Started from February 2021 based on daily observation analysis (697 rows/day avg)
- Clean month start with high data density
- Matches actual data availability in BigQuery

### 2. BigQuery Client (src/data/bigquery_client.py)
**Updated methods:**

#### `__init__()` (lines 19-34)
- Added `sentiment_table` parameter with default `"posts_comments_enhanced_sentiment"`
- Stores `self.sentiment_table` for use in queries
- Allows flexible table selection via environment variable

#### `get_post_sentiment_aggregation()` (lines 284-329)
**MAJOR CHANGE: Daily → 5-minute aggregation**
- Aggregation interval: **Daily (DATE) → 5-minute bins (TIMESTAMP)**
- Implementation: `TIMESTAMP_SECONDS(CAST(FLOOR(UNIX_SECONDS(post_created_utc) / 300) * 300 AS INT64))`
- Table: Uses `self.sentiment_table` from .env
- Columns:
  - `comment_sentiment_label` → `new_sentiment_label`
  - `comment_sentiment_score` → `new_sentiment_compound`
  - Added `new_sentiment_positive`, `new_sentiment_negative`, `new_sentiment_neutral`
  - Added `avg_compound_sentiment` to output
- Note: `avg_post_score` and `avg_comment_score` set to 0.0 (not available in enhanced table)

**Impact:**
- 5-minute granularity: 27,546 rows (vs 10,355 daily)
- Compression: 20.6x (568K raw → 27K aggregated)
- Avg 20.6 comments per 5-minute bin
- Enables time-series analysis at crypto-relevant timescales

### 3. Hierarchical Data Processor (src/data/hierarchical_data_processor.py)
**New file copied from `info_spillover/` with corrections:**

#### Sentiment validation (lines 78-120)
- Fixed VADER compound score range: **[0, 1] → [-1, +1]**
- Updated anomaly detection for neutral scores: `< 0.6` → `|score| > 0.3`
- Renamed metric: `neutral_low_confidence` → `neutral_extreme_scores`

#### Raw data query (lines 515-530)
- Table: `posts_comments` → `posts_comments_enhanced_sentiment`
- Columns: `comment_sentiment_label` → `new_sentiment_label`
- Columns: `comment_sentiment_score` → `new_sentiment_compound`
- Set `score` to 0 placeholder (not available)

#### Fear/Greed/FOMO scores (lines 213-216)
Updated thresholds for [-1, +1] range:
- Fear: `sentiment_score > 0.8` → `sentiment_score < -0.5`
- Greed: `sentiment_score > 0.8` → `sentiment_score > 0.5`
- FOMO: `sentiment_score < 0.7` → `0.2 < sentiment_score < 0.5`

### 4. Test Results

#### Data Loading Summary
```
Total rows loaded: 27,546 5-minute aggregates
Date range: 2021-02-01 07:00:00 to 2025-10-07 08:45:00
Unique subreddits: 21
Time span: 1,709 days (4.7 years)

Raw data compression:
- Raw comments: 568,799
- 5-min aggregates: 27,546
- Compression ratio: 20.6x
- Avg comments per bin: 20.6

Top subreddits (5-min bins):
- CryptoCurrency: 1,960 bins
- cryptomoonshots: 1,820 bins
- BitcoinBeginners: 1,783 bins
- defi: 1,695 bins
- cardano: 1,641 bins
```

#### Sentiment Distribution
```
Label distribution (5-min aggregates):
- positive: 10,014 (36.4%)
- negative:  9,709 (35.3%)
- neutral:   7,823 (28.4%)

Score stats (VADER compound):
- mean: 0.046
- std: 0.352
- min: -0.941
- max: 0.987
- median: 0.010
- Valid range: [-1, +1] ✅
```

#### Quality Report
✅ All validation checks passed:
- Timestamp validation: 27,546 records, 1,709 days span
- Temporal granularity: 5-minute bins
- Avg gap: 88.96 minutes (sparse activity expected)
- Sentiment score range: Valid [-1, +1]
- Subreddit coverage: 21 subreddits
- No extreme neutral scores (neutral_extreme_scores: 0)

## Files Modified
1. `.env` - Date range and format corrections
2. `src/data/bigquery_client.py` - Enhanced sentiment table migration
3. `src/data/hierarchical_data_processor.py` - New file with corrections

## Files Created
1. `test_data_processor.py` - Verification test script
2. `CHANGELOG.md` - This file

## Next Steps

### Immediate
- [ ] Verify price data integration (5-minute granularity)
- [ ] Test VAR dataset creation
- [ ] Test network feature extraction

### Pipeline Components (to be copied)
From `info_spillover/`:
- [ ] `src/data/var_dataset_processor.py` - VAR-specific transformations
- [ ] `src/analysis/diebold_yilmaz_spillover.py` - Spillover analysis
- [ ] `src/main_pipeline.py` - Main orchestrator

### Documentation
- [ ] Update README with new data sources
- [ ] Document VADER compound sentiment interpretation
- [ ] Create data flow diagram

## Verification Commands

```bash
# Run data loading test
python3 test_data_processor.py

# Check BigQuery data availability
python3 -c "
from google.cloud import bigquery
client = bigquery.Client(project='informationspillover')
query = '''
SELECT MIN(DATE(post_created_utc)) as min_date,
       MAX(DATE(post_created_utc)) as max_date,
       COUNT(*) as total_rows
FROM \`informationspillover.spillover_statistical_test.posts_comments_enhanced_sentiment\`
'''
print(client.query(query).to_dataframe())
"

# Verify sentiment score range
python3 -c "
from google.cloud import bigquery
client = bigquery.Client(project='informationspillover')
query = '''
SELECT MIN(new_sentiment_compound) as min_score,
       MAX(new_sentiment_compound) as max_score,
       AVG(new_sentiment_compound) as avg_score
FROM \`informationspillover.spillover_statistical_test.posts_comments_enhanced_sentiment\`
'''
print(client.query(query).to_dataframe())
"
```

## Migration Notes

### Why HierarchicalDataProcessor?
The original `prepare_data.py` (160 lines) was a simple helper script with basic train/val/test splits. The production pipeline uses:
- `main_pipeline.py` (1907 lines) - orchestrator
- `HierarchicalDataProcessor` - advanced data processing with:
  - Data quality validation
  - VAR dataset preparation
  - Network construction
  - Granger causality testing
  - Temporal aggregation

### VADER Compound Sentiment
Range: **[-1, +1]** where:
- -1.0 = extremely negative
- 0.0 = neutral
- +1.0 = extremely positive

**NOT** the old [0, 1] range that was incorrectly assumed in original code.

### Data Density Analysis
Chosen start date (2021-02-01) based on:
- First month with 500+ rows/day: Feb 2021 (697 avg)
- Clean month start vs mid-month
- Total dataset: 568,799 rows from 2021-02-01 to 2025-10-07

### Architecture Decision
**Full pipeline approach** (VAR + Hierarchical + Spillover) but implementing step-by-step:
1. ✅ Data processing layer (this update)
2. ⏳ VAR analysis layer
3. ⏳ Spillover analysis layer
4. ⏳ Hierarchical modeling layer

---

**Date:** 2026-01-11
**Status:** ✅ Data processing layer verified and working
**Next:** VAR dataset processor migration
