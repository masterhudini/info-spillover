# Modeling Implementation Plan (info_spillover_enhanced)

## Objective
Implement the modeling stage in the new project so it is fully consistent with the DY pipeline: fixed subreddit selection, strict diagnostics, and no silent fallbacks.

## Scope
- Inputs come from `HierarchicalDataProcessor` output and DY spillover network.
- Modeling should consume the same subreddit set used in DY (`SUBREDDITS_SELECTED`).
- No fallback behaviors: if required artifacts are missing or inconsistent, fail fast.

## Required Inputs
- Processed dataframe from `HierarchicalDataProcessor.process_full_pipeline()`
- DY spillover network and metrics from `build_network_features`
- Environment-driven configuration:
  - `SUBREDDITS_SELECTED`
  - `DY_RESAMPLE_FREQ`, `DY_MAX_LAGS`, `DY_FORECAST_HORIZON`
  - `DY_DIFF_SUBREDDITS`, `DY_FFILL_LIMIT`
  - Price settings: `PRICE_MERGE_TOLERANCE`
  - Any modeling hyperparams (new env vars or config file)

## Step 1: Define modeling data contract
- Create a clear schema for the modeling dataset:
  - Required columns: `subreddit`, `post_created_utc`, `compound_sentiment`, `next_sentiment`, `sentiment_change`, `next_return`, `return_direction`, DY network features (`net_spillover`, `pagerank`, `in_degree`, `out_degree`, `clustering`).
- Validation checks:
  - All subreddits in data must be exactly `SUBREDDITS_SELECTED`.
  - DY network nodes must match `SUBREDDITS_SELECTED`.
  - No missing values for required columns after final cleanup.

## Step 2: Build a modeling data module
- Location: `src/modules/data/` (new module).
- Responsibilities:
  - Filter to `SUBREDDITS_SELECTED` (no fallback).
  - Temporal ordering, consistent frequency checks.
  - Train/val/test split based on time (no random split).
  - Build time-indexed node features for GNN input (static FEVD graph).

## Step 3: Model architecture
- DY-consistent architecture:
  - MLP encoder for node features (no LSTM/GRU).
  - Mandatory GNN using static FEVD spillover weights.
- Strict requirements:
  - If DY network has zero edges, stop (no fallback self-loops).
  - Edge weights are FEVD spillover weights (no learned rewiring).

## Step 4: Training pipeline
- Implement a minimal training harness:
  - Batch creation, loss definition, evaluation metrics.
  - Log results to file and/or MLflow (optional).
- Training should fail if:
  - Any subreddit dataset is empty.
  - Network nodes mismatch dataset subreddits.

## Step 5: Evaluation
- Metrics to report:
  - Return direction classification: accuracy/F1 on `return_direction`.
- Report per-subreddit metrics and overall aggregate.

## Step 6: Integration points
- Add a modeling entry-point that takes:
  - `processed_data` + `spillover_network`
  - `model_config` (env or yaml)
- Return:
  - Trained model(s)
  - Metrics summary
  - Artifacts (optional)

## No-fallback policy
- No auto-selection of columns or subreddits.
- No filling missing network edges.
- Any missing required artifact should raise an error.

## Deliverables
1. `src/modules/data/` with a `ModelingDataModule` (time-split datasets).
2. `src/modules/models/` with MLP encoder + mandatory GNN (static FEVD graph).
3. Training entry point (script or function) with strict validations.
4. Basic evaluation report (JSON/CSV).

## Decisions Locked
- GNN is mandatory and uses static FEVD weights.
- Target is `return_direction` only.
