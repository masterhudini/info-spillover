"""
Transform Validator - validates data transformations in the pipeline.
Each transformation should have a specific validation to ensure data integrity.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class TransformValidationError(Exception):
    """Raised when a transformation validation fails."""
    pass


class TransformValidator:
    """
    Validates data transformations to ensure integrity throughout the pipeline.

    Each validation method checks specific invariants that should hold after
    a transformation. Failures raise TransformValidationError with details.
    """

    def __init__(self, strict: bool = True, log_warnings: bool = True):
        """
        Args:
            strict: If True, raise errors on validation failures. If False, log warnings.
            log_warnings: If True, log warnings for non-critical issues.
        """
        self.strict = strict
        self.log_warnings = log_warnings
        self.validation_log: List[Dict[str, Any]] = []

    def _fail(self, message: str, context: Dict[str, Any] = None):
        """Handle validation failure based on strict mode."""
        log_entry = {
            'message': message,
            'context': context or {},
            'failed': True
        }
        self.validation_log.append(log_entry)

        if self.strict:
            raise TransformValidationError(message)
        else:
            logger.warning(f"Validation warning: {message}")

    def _pass(self, message: str, context: Dict[str, Any] = None):
        """Log successful validation."""
        log_entry = {
            'message': message,
            'context': context or {},
            'failed': False
        }
        self.validation_log.append(log_entry)
        logger.debug(f"Validation passed: {message}")

    # =========================================================================
    # MERGE VALIDATIONS
    # =========================================================================

    def validate_merge_row_count(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        operation: str = "merge",
        allow_fewer: bool = False,
        max_loss_pct: float = 0.0
    ):
        """
        Validate that merge didn't unexpectedly change row count.

        Args:
            df_before: DataFrame before merge
            df_after: DataFrame after merge
            operation: Name of operation for error message
            allow_fewer: If True, allow fewer rows (for inner joins)
            max_loss_pct: Maximum allowed percentage of row loss (0.0 - 1.0)
        """
        before_count = len(df_before)
        after_count = len(df_after)

        if before_count == 0:
            self._fail(f"{operation}: input DataFrame is empty")
            return

        loss_pct = (before_count - after_count) / before_count

        if after_count > before_count:
            self._fail(
                f"{operation}: row count increased from {before_count} to {after_count}. "
                f"This suggests duplicate matches or incorrect join type.",
                {'before': before_count, 'after': after_count, 'increase': after_count - before_count}
            )
        elif after_count < before_count:
            if not allow_fewer:
                self._fail(
                    f"{operation}: row count decreased from {before_count} to {after_count} "
                    f"({loss_pct:.1%} loss). Expected equal row count.",
                    {'before': before_count, 'after': after_count, 'loss_pct': loss_pct}
                )
            elif loss_pct > max_loss_pct:
                self._fail(
                    f"{operation}: row loss {loss_pct:.1%} exceeds max allowed {max_loss_pct:.1%}. "
                    f"Before: {before_count}, After: {after_count}",
                    {'before': before_count, 'after': after_count, 'loss_pct': loss_pct}
                )
            else:
                self._pass(
                    f"{operation}: row count decreased within allowed limit "
                    f"({loss_pct:.1%} <= {max_loss_pct:.1%})",
                    {'before': before_count, 'after': after_count}
                )
        else:
            self._pass(f"{operation}: row count preserved ({before_count} rows)")

    def validate_merge_asof_coverage(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        merge_key: str,
        tolerance_col: str = None,
        max_unmatched_pct: float = 0.05
    ):
        """
        Validate merge_asof matched most records within tolerance.

        Args:
            df_before: DataFrame before merge (left side)
            df_after: DataFrame after merge
            merge_key: The column that was merged on from right side
            tolerance_col: Column from right side to check for NaN (unmatched)
            max_unmatched_pct: Maximum allowed percentage of unmatched records
        """
        if tolerance_col and tolerance_col in df_after.columns:
            unmatched = df_after[tolerance_col].isna().sum()
            total = len(df_after)
            unmatched_pct = unmatched / total if total > 0 else 0

            if unmatched_pct > max_unmatched_pct:
                self._fail(
                    f"merge_asof: {unmatched_pct:.1%} records unmatched "
                    f"(>{max_unmatched_pct:.1%} threshold). "
                    f"Check tolerance or data alignment.",
                    {'unmatched': unmatched, 'total': total, 'unmatched_pct': unmatched_pct}
                )
            else:
                self._pass(
                    f"merge_asof: {100-unmatched_pct*100:.1f}% records matched",
                    {'matched_pct': 1 - unmatched_pct}
                )

    def validate_merge_by_column(
        self,
        df: pd.DataFrame,
        left_col: str,
        right_col: str,
        operation: str = "merge"
    ):
        """
        Validate that merge was done with proper grouping (by= parameter).

        After merge_asof with by=, values in left_col and right_col should match.
        """
        if left_col not in df.columns or right_col not in df.columns:
            self._fail(
                f"{operation}: Cannot validate grouping - columns missing. "
                f"left_col={left_col} in df: {left_col in df.columns}, "
                f"right_col={right_col} in df: {right_col in df.columns}"
            )
            return

        # Check where both are not null
        mask = df[left_col].notna() & df[right_col].notna()
        mismatches = (df.loc[mask, left_col] != df.loc[mask, right_col]).sum()

        if mismatches > 0:
            self._fail(
                f"{operation}: {mismatches} rows have mismatched {left_col} vs {right_col}. "
                f"This indicates merge_asof is missing 'by=' parameter.",
                {'mismatches': mismatches, 'total_compared': mask.sum()}
            )
        else:
            self._pass(f"{operation}: grouping column values match correctly")

    # =========================================================================
    # LOG TRANSFORM VALIDATIONS
    # =========================================================================

    def validate_log_transform(
        self,
        series: pd.Series,
        name: str,
        allow_nan: bool = False,
        original_min: float = None
    ):
        """
        Validate log/log1p transform result.

        Args:
            series: Transformed series
            name: Column name for error messages
            allow_nan: If True, NaN values are acceptable
            original_min: If provided, check that log1p was used for values that could be 0
        """
        if series.empty:
            self._fail(f"log_transform({name}): result is empty")
            return

        # Check for -inf (from log(0))
        neg_inf_count = np.isneginf(series).sum()
        if neg_inf_count > 0:
            self._fail(
                f"log_transform({name}): {neg_inf_count} -inf values detected. "
                f"Use log1p() instead of log() for data that can be 0.",
                {'neg_inf_count': neg_inf_count}
            )

        # Check for +inf
        pos_inf_count = np.isposinf(series).sum()
        if pos_inf_count > 0:
            self._fail(
                f"log_transform({name}): {pos_inf_count} +inf values detected. "
                f"Input may have extremely large values.",
                {'pos_inf_count': pos_inf_count}
            )

        # Check for unexpected NaN
        nan_count = series.isna().sum()
        if nan_count > 0 and not allow_nan:
            self._fail(
                f"log_transform({name}): {nan_count} NaN values produced. "
                f"Check for negative inputs to log().",
                {'nan_count': nan_count}
            )

        # For log1p, result should be >= 0 if input was >= 0
        if original_min is not None and original_min >= 0:
            if (series < 0).any():
                self._fail(
                    f"log_transform({name}): negative values in output but input min was {original_min}",
                    {'output_min': series.min()}
                )

        self._pass(f"log_transform({name}): valid (range [{series.min():.4f}, {series.max():.4f}])")

    # =========================================================================
    # DIFF VALIDATIONS
    # =========================================================================

    def validate_diff(
        self,
        series_before: pd.Series,
        series_after: pd.Series,
        name: str,
        expected_nan_count: int = 1
    ):
        """
        Validate diff() transform.

        Args:
            series_before: Original series
            series_after: Diffed series
            name: Column name for error messages
            expected_nan_count: Expected NaN count from diff (usually 1 for first element)
        """
        nan_before = series_before.isna().sum()
        nan_after = series_after.isna().sum()
        new_nans = nan_after - nan_before

        if new_nans != expected_nan_count:
            self._fail(
                f"diff({name}): expected {expected_nan_count} new NaN, got {new_nans}. "
                f"Before: {nan_before} NaN, After: {nan_after} NaN",
                {'expected_new_nan': expected_nan_count, 'actual_new_nan': new_nans}
            )
        else:
            self._pass(f"diff({name}): correct NaN count after diff")

        # Check lengths match
        if len(series_before) != len(series_after):
            self._fail(
                f"diff({name}): length changed from {len(series_before)} to {len(series_after)}",
                {'before_len': len(series_before), 'after_len': len(series_after)}
            )

    # =========================================================================
    # SHIFT VALIDATIONS
    # =========================================================================

    def validate_shift(
        self,
        series_before: pd.Series,
        series_after: pd.Series,
        name: str,
        shift_periods: int
    ):
        """
        Validate shift() transform.

        Args:
            series_before: Original series
            series_after: Shifted series
            name: Column name
            shift_periods: Number of periods shifted (negative = backward)
        """
        nan_before = series_before.isna().sum()
        nan_after = series_after.isna().sum()
        expected_new_nan = abs(shift_periods)
        new_nans = nan_after - nan_before

        if new_nans != expected_new_nan:
            self._fail(
                f"shift({name}, {shift_periods}): expected {expected_new_nan} new NaN, got {new_nans}",
                {'expected': expected_new_nan, 'actual': new_nans}
            )
        else:
            self._pass(f"shift({name}): correct NaN count after shift({shift_periods})")

    # =========================================================================
    # RESAMPLE VALIDATIONS
    # =========================================================================

    def validate_resample(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        freq: str,
        time_col: str,
        min_coverage_pct: float = 0.5,
        check_gaps: bool = True
    ):
        """
        Validate resample operation.

        Args:
            df_before: DataFrame before resample
            df_after: DataFrame after resample
            freq: Resample frequency
            time_col: Timestamp column name
            min_coverage_pct: Minimum expected data coverage (0.0 - 1.0)
            check_gaps: If True, warn about large gaps
        """
        if df_after.empty:
            self._fail("resample: output is empty")
            return

        # Calculate expected vs actual periods
        if time_col in df_before.columns:
            time_range = df_before[time_col].max() - df_before[time_col].min()
        elif df_before.index.name == time_col or isinstance(df_before.index, pd.DatetimeIndex):
            time_range = df_before.index.max() - df_before.index.min()
        else:
            self._pass("resample: cannot determine time range for coverage check")
            return

        # Actual periods
        actual_periods = len(df_after)

        # Log result
        self._pass(
            f"resample({freq}): {actual_periods} periods created from {len(df_before)} records",
            {'before_rows': len(df_before), 'after_rows': actual_periods, 'freq': freq}
        )

    # =========================================================================
    # DROPNA VALIDATIONS
    # =========================================================================

    def validate_dropna(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        subset: List[str],
        max_drop_pct: float = 0.1,
        operation: str = "dropna"
    ):
        """
        Validate dropna didn't remove too many rows.

        Args:
            df_before: DataFrame before dropna
            df_after: DataFrame after dropna
            subset: Columns checked for NA
            max_drop_pct: Maximum allowed percentage of dropped rows
            operation: Operation name for logging
        """
        before_count = len(df_before)
        after_count = len(df_after)
        dropped = before_count - after_count
        drop_pct = dropped / before_count if before_count > 0 else 0

        if drop_pct > max_drop_pct:
            self._fail(
                f"{operation}(subset={subset}): dropped {drop_pct:.1%} rows "
                f"(>{max_drop_pct:.1%} threshold). "
                f"Before: {before_count}, After: {after_count}, Dropped: {dropped}",
                {'dropped': dropped, 'drop_pct': drop_pct, 'subset': subset}
            )
        elif dropped > 0 and self.log_warnings:
            logger.info(
                f"{operation}: dropped {dropped} rows ({drop_pct:.1%}) with NA in {subset}"
            )
            self._pass(f"{operation}: dropped {dropped} rows ({drop_pct:.1%})")
        else:
            self._pass(f"{operation}: no rows dropped")

    # =========================================================================
    # GROUPBY TRANSFORM VALIDATIONS
    # =========================================================================

    def validate_groupby_transform(
        self,
        df_before: pd.DataFrame,
        series_after: pd.Series,
        group_col: str,
        transform_name: str
    ):
        """
        Validate groupby().transform() preserved row count and index.

        Args:
            df_before: Original DataFrame
            series_after: Result of groupby transform
            group_col: Column used for grouping
            transform_name: Name of transform for logging
        """
        if len(df_before) != len(series_after):
            self._fail(
                f"groupby({group_col}).transform({transform_name}): "
                f"length changed from {len(df_before)} to {len(series_after)}",
                {'before': len(df_before), 'after': len(series_after)}
            )
        else:
            self._pass(f"groupby.transform({transform_name}): length preserved")

    # =========================================================================
    # TYPE CONVERSION VALIDATIONS
    # =========================================================================

    def validate_astype(
        self,
        series_before: pd.Series,
        series_after: pd.Series,
        name: str,
        target_dtype: str
    ):
        """
        Validate type conversion didn't introduce NaN or change values.

        Args:
            series_before: Original series
            series_after: Converted series
            name: Column name
            target_dtype: Target dtype string
        """
        nan_before = series_before.isna().sum()
        nan_after = series_after.isna().sum()

        if nan_after > nan_before:
            new_nans = nan_after - nan_before
            self._fail(
                f"astype({name}, {target_dtype}): introduced {new_nans} new NaN values. "
                f"Some values couldn't be converted.",
                {'new_nans': new_nans}
            )

        # For numeric conversions, check value preservation
        if np.issubdtype(series_after.dtype, np.number):
            # Compare non-null values
            mask = series_before.notna() & series_after.notna()
            if mask.any():
                # Allow small floating point differences
                diff = (series_before[mask].astype(float) - series_after[mask].astype(float)).abs()
                max_diff = diff.max()
                if max_diff > 1e-6:
                    self._fail(
                        f"astype({name}): max value difference {max_diff} after conversion",
                        {'max_diff': max_diff}
                    )

        self._pass(f"astype({name}, {target_dtype}): conversion successful")

    # =========================================================================
    # ROLLING VALIDATIONS
    # =========================================================================

    def validate_rolling(
        self,
        series_before: pd.Series,
        series_after: pd.Series,
        name: str,
        window: int,
        min_periods: int = None
    ):
        """
        Validate rolling window operation.

        Args:
            series_before: Original series
            series_after: Result of rolling operation
            name: Column name
            window: Rolling window size
            min_periods: Minimum periods parameter (default=window)
        """
        if min_periods is None:
            min_periods = window

        if len(series_before) != len(series_after):
            self._fail(
                f"rolling({name}): length changed from {len(series_before)} to {len(series_after)}"
            )
            return

        # Check expected NaN at start
        expected_initial_nan = min_periods - 1
        initial_nan = series_after.iloc[:expected_initial_nan].isna().sum() if expected_initial_nan > 0 else 0

        # After initial period, check for unexpected NaN
        if len(series_after) > expected_initial_nan:
            rest = series_after.iloc[expected_initial_nan:]
            rest_nan = rest.isna().sum()
            input_nan = series_before.iloc[expected_initial_nan:].isna().sum()

            # Allow for NaN propagation from input
            if rest_nan > input_nan * window:  # rough heuristic
                if self.log_warnings:
                    logger.warning(
                        f"rolling({name}): {rest_nan} NaN values after initial window "
                        f"(input had {input_nan} NaN)"
                    )

        self._pass(f"rolling({name}, window={window}): computed successfully")

    # =========================================================================
    # VALUE RANGE VALIDATIONS
    # =========================================================================

    def validate_range(
        self,
        series: pd.Series,
        name: str,
        min_val: float = None,
        max_val: float = None,
        allow_nan: bool = True
    ):
        """
        Validate values are within expected range.

        Args:
            series: Series to validate
            name: Column name
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            allow_nan: If False, fail on any NaN
        """
        if not allow_nan and series.isna().any():
            self._fail(
                f"range({name}): contains {series.isna().sum()} NaN values",
                {'nan_count': series.isna().sum()}
            )

        clean = series.dropna()
        if clean.empty:
            self._fail(f"range({name}): all values are NaN")
            return

        actual_min = clean.min()
        actual_max = clean.max()

        if min_val is not None and actual_min < min_val:
            self._fail(
                f"range({name}): min value {actual_min} < allowed {min_val}",
                {'actual_min': actual_min, 'allowed_min': min_val}
            )

        if max_val is not None and actual_max > max_val:
            self._fail(
                f"range({name}): max value {actual_max} > allowed {max_val}",
                {'actual_max': actual_max, 'allowed_max': max_val}
            )

        self._pass(
            f"range({name}): values in [{actual_min:.4f}, {actual_max:.4f}]",
            {'min': actual_min, 'max': actual_max}
        )

    # =========================================================================
    # SENTIMENT-SPECIFIC VALIDATIONS
    # =========================================================================

    def validate_sentiment_score(self, series: pd.Series, name: str = "sentiment"):
        """Validate VADER-style sentiment scores are in [-1, 1]."""
        self.validate_range(series, name, min_val=-1.0, max_val=1.0)

    # =========================================================================
    # PRICE DATA VALIDATIONS
    # =========================================================================

    def validate_price_data(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        symbol_col: str = 'symbol',
        time_col: str = 'snapped_at'
    ):
        """
        Validate price data integrity.

        Args:
            df: Price DataFrame
            price_col: Price column name
            symbol_col: Symbol column name
            time_col: Timestamp column name
        """
        # Check for negative prices
        if (df[price_col] <= 0).any():
            bad_count = (df[price_col] <= 0).sum()
            self._fail(
                f"price_data: {bad_count} non-positive prices detected",
                {'non_positive_count': bad_count}
            )

        # Check for gaps per symbol
        for symbol in df[symbol_col].unique():
            symbol_data = df[df[symbol_col] == symbol].sort_values(time_col)
            if len(symbol_data) < 2:
                continue

            gaps = symbol_data[time_col].diff().dropna()
            median_gap = gaps.median()
            large_gaps = (gaps > median_gap * 10).sum()

            if large_gaps > 0 and self.log_warnings:
                logger.warning(
                    f"price_data({symbol}): {large_gaps} gaps >10x median gap detected"
                )

        self._pass(f"price_data: {len(df)} records for {df[symbol_col].nunique()} symbols")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        failures = [v for v in self.validation_log if v['failed']]
        passes = [v for v in self.validation_log if not v['failed']]

        return {
            'total_validations': len(self.validation_log),
            'passed': len(passes),
            'failed': len(failures),
            'failure_messages': [f['message'] for f in failures],
            'all_validations': self.validation_log
        }

    def reset(self):
        """Reset validation log."""
        self.validation_log = []
