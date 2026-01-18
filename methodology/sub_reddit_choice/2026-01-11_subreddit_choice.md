# Subreddit selection for DY spillover analysis (2026-01-11)

## Goal
Select a fixed set of subreddits for Diebold-Yilmaz (VAR/FEVD) spillover analysis using the available 5-minute aggregated sentiment data.

## Data basis
- Source: BigQuery aggregated sentiment table (`posts_comments_enhanced_sentiment`).
- Time range: `2021-02-01` to `2025-10-07`.
- Aggregation: 5-minute bins (27,546 rows, 21 subreddits).

## Selection method
1. Rank subreddits by number of available 5-minute bins (data coverage).
2. Evaluate overlap of common timestamps for different resample frequencies and top-N lists.
3. Choose a fixed list that maximizes coverage and keeps the set stable for DY/VAR comparability.

## Top-10 by coverage (chosen)
Based on row counts (5-minute bins):
- CryptoCurrency
- cryptomoonshots
- BitcoinBeginners
- defi
- cardano
- ethtrader
- Bitcoin
- solana
- CryptoMarkets
- ethereum

This list is stored in `.env` as `SUBREDDITS_SELECTED` and used directly in queries.

## Overlap summary (common timestamps across selected subreddits)
Key results from the overlap scan:
- 5T / 15T / 30T: 0 common timestamps for top-10.
- 1H: 0 common timestamps for top-10.
- 2H: 2 common timestamps for top-10.
- 4H: 9 common timestamps for top-10.
- 6H: 22 common timestamps for top-10.
- 12H: 69 common timestamps for top-10.
- 1D: 96 common timestamps for top-10.

## Implication for DY (VAR/FEVD)
- DY requires a fully observed panel (no missing values per timestamp across all selected subreddits).
- With top-10 and 2H frequency, the fully observed panel is effectively empty unless forward-fill (imputation) is enabled.
- If strict no-imputation is required, either the frequency must be coarser (>=12H) or the subreddit set must be reduced.

## Decision rationale
We keep the top-10 list for interpretability and coverage consistency. When using a 2H DY frequency, forward-fill is required to construct the panel. If strict no-imputation is mandated, use 12H or 1D with this list, or reduce to top-5 for 6H/12H.
