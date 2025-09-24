"""
Enhanced MLflow Tracker with Comprehensive Statistical Validation

This module extends the basic MLflow integration to include:
1. Comprehensive statistical validation tracking
2. P-values and significance testing results
3. Model assumption testing
4. Spillover analysis validation
5. Automated statistical reporting

Integrates with the StatisticalValidationFramework to provide
complete experimental tracking and reproducibility.
"""

import mlflow
import mlflow.sklearn
import mlflow.statsmodels
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .statistical_validation import StatisticalValidationFramework, StatisticalTest, TestResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMLflowTracker:
    """
    Enhanced MLflow tracker with comprehensive statistical validation
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///enhanced_mlflow.db"):
        """
        Initialize enhanced MLflow tracker

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Initialize statistical validation framework
        self.validator = StatisticalValidationFramework()

        # Tracking state
        self.current_run = None
        self.validation_results = {}

        logger.info(f"Enhanced MLflow tracker initialized: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None, run_id: Optional[str] = None) -> str:
        """
        Start MLflow run with enhanced tracking capabilities

        Args:
            run_name: Optional run name
            run_id: Optional existing run ID to continue

        Returns:
            Run ID
        """
        if run_id:
            self.current_run = mlflow.start_run(run_id=run_id)
        else:
            self.current_run = mlflow.start_run(run_name=run_name)

        run_id = self.current_run.info.run_id

        # Log framework metadata
        mlflow.set_tags({
            "framework_version": "enhanced_v1.0",
            "statistical_validation": "enabled",
            "start_time": datetime.now().isoformat()
        })

        logger.info(f"MLflow run started: {run_id}")
        return run_id

    def log_var_validation(self, data: pd.DataFrame, max_lags: int = 10) -> Dict[str, StatisticalTest]:
        """
        Perform and log comprehensive VAR model validation

        Args:
            data: Time series data for VAR validation
            max_lags: Maximum lags for VAR testing

        Returns:
            Dictionary of validation test results
        """
        logger.info("🔍 Performing VAR model validation...")

        # Perform validation
        var_results = self.validator.validate_var_assumptions(data, max_lags)

        # Log to MLflow
        self._log_statistical_tests(var_results, category="var_validation")

        # Log summary metrics
        var_summary = self._calculate_test_summary(var_results)
        mlflow.log_metrics({
            "var_tests_total": var_summary['total'],
            "var_tests_passed": var_summary['passed'],
            "var_tests_failed": var_summary['failed'],
            "var_pass_rate": var_summary['pass_rate'],
            "var_model_valid": 1 if var_summary['pass_rate'] > 0.7 else 0
        })

        # Log detailed results as artifact
        self._save_validation_artifact(var_results, "var_validation_results.json")

        self.validation_results['var_validation'] = var_results
        logger.info(f"✅ VAR validation completed: {var_summary['pass_rate']:.2%} pass rate")

        return var_results

    def log_spillover_validation(self, spillover_data: pd.DataFrame,
                               event_dates: Optional[List[str]] = None) -> Dict[str, StatisticalTest]:
        """
        Perform and log spillover significance testing

        Args:
            spillover_data: Spillover time series data
            event_dates: Optional list of event dates for testing

        Returns:
            Dictionary of spillover test results
        """
        logger.info("🔍 Performing spillover significance testing...")

        # Perform validation
        spillover_results = self.validator.test_spillover_significance(spillover_data, event_dates)

        # Log to MLflow
        self._log_statistical_tests(spillover_results, category="spillover_validation")

        # Log summary metrics
        spillover_summary = self._calculate_test_summary(spillover_results)
        mlflow.log_metrics({
            "spillover_tests_total": spillover_summary['total'],
            "spillover_tests_passed": spillover_summary['passed'],
            "spillover_tests_failed": spillover_summary['failed'],
            "spillover_pass_rate": spillover_summary['pass_rate'],
            "spillover_significant": 1 if spillover_summary['pass_rate'] > 0.5 else 0
        })

        # Log spillover statistics
        if not spillover_data.empty:
            for column in spillover_data.columns:
                if spillover_data[column].notna().sum() > 0:
                    series = spillover_data[column].dropna()
                    mlflow.log_metrics({
                        f"spillover_{column}_mean": float(series.mean()),
                        f"spillover_{column}_std": float(series.std()),
                        f"spillover_{column}_min": float(series.min()),
                        f"spillover_{column}_max": float(series.max()),
                        f"spillover_{column}_volatility": float(series.std() / series.mean()) if series.mean() != 0 else 0
                    })

        # Save detailed results
        self._save_validation_artifact(spillover_results, "spillover_validation_results.json")

        self.validation_results['spillover_validation'] = spillover_results
        logger.info(f"✅ Spillover validation completed: {spillover_summary['pass_rate']:.2%} pass rate")

        return spillover_results

    def log_ml_validation(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any],
                         cv_method: str = 'timeseries') -> Dict[str, StatisticalTest]:
        """
        Perform and log comprehensive ML model validation

        Args:
            X: Feature matrix
            y: Target variable
            models: Dictionary of models to validate
            cv_method: Cross-validation method

        Returns:
            Dictionary of ML validation test results
        """
        logger.info("🔍 Performing ML model validation...")

        # Perform validation
        ml_results = self.validator.validate_ml_models(X, y, models, cv_method)

        # Log to MLflow
        self._log_statistical_tests(ml_results, category="ml_validation")

        # Log summary metrics
        ml_summary = self._calculate_test_summary(ml_results)
        mlflow.log_metrics({
            "ml_tests_total": ml_summary['total'],
            "ml_tests_passed": ml_summary['passed'],
            "ml_tests_failed": ml_summary['failed'],
            "ml_pass_rate": ml_summary['pass_rate'],
            "ml_models_reliable": 1 if ml_summary['pass_rate'] > 0.6 else 0
        })

        # Log feature importance and model comparison
        self._log_model_comparison_metrics(ml_results, models)

        # Save detailed results
        self._save_validation_artifact(ml_results, "ml_validation_results.json")

        self.validation_results['ml_validation'] = ml_results
        logger.info(f"✅ ML validation completed: {ml_summary['pass_rate']:.2%} pass rate")

        return ml_results

    def log_comprehensive_validation(self, data: pd.DataFrame, spillover_data: pd.DataFrame,
                                   X: pd.DataFrame, y: pd.Series, models: Dict[str, Any],
                                   event_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform and log complete statistical validation pipeline

        Args:
            data: Raw time series data for VAR validation
            spillover_data: Spillover indices data
            X: ML features
            y: ML target
            models: ML models to validate
            event_dates: Optional event dates for spillover testing

        Returns:
            Complete validation report
        """
        logger.info("🚀 Running comprehensive statistical validation pipeline...")

        # Perform all validations
        var_results = self.log_var_validation(data)
        spillover_results = self.log_spillover_validation(spillover_data, event_dates)
        ml_results = self.log_ml_validation(X, y, models)

        # Generate comprehensive report
        validation_report = self.validator.generate_validation_report()

        # Log overall validation metrics
        mlflow.log_metrics({
            "overall_validation_score": validation_report['summary']['tests_passed'] /
                                      max(1, validation_report['summary']['total_tests_performed']),
            "total_statistical_tests": validation_report['summary']['total_tests_performed'],
            "tests_passed": validation_report['summary']['tests_passed'],
            "tests_failed": validation_report['summary']['tests_failed'],
            "validation_reliability": self._calculate_reliability_score(validation_report)
        })

        # Log validation parameters
        mlflow.log_params({
            "significance_level": self.validator.alpha,
            "bootstrap_iterations": self.validator.bootstrap_iterations,
            "validation_framework": "comprehensive_v1.0"
        })

        # Save comprehensive report
        self._save_validation_artifact(validation_report, "comprehensive_validation_report.json")

        # Generate and save markdown report
        markdown_report = self._generate_markdown_report(validation_report)
        self._save_text_artifact(markdown_report, "validation_report.md")

        logger.info("✅ Comprehensive validation completed and logged")
        return validation_report

    def log_model_with_validation(self, model: Any, model_name: str,
                                validation_results: Optional[Dict] = None,
                                signature: Optional[Any] = None,
                                input_example: Optional[Any] = None) -> None:
        """
        Log model with associated validation results

        Args:
            model: Trained model object
            model_name: Name for the model
            validation_results: Optional validation results to associate
            signature: Optional MLflow model signature
            input_example: Optional input example
        """
        # Log the model
        mlflow.sklearn.log_model(
            model,
            model_name,
            signature=signature,
            input_example=input_example
        )

        # Log validation status
        if validation_results:
            model_valid = all(
                result.result in [TestResult.PASS, TestResult.WARNING]
                for result in validation_results.values()
                if 'permutation' in result.name.lower() or 'best_model' in result.name.lower()
            )

            mlflow.log_metrics({
                f"{model_name}_validation_passed": 1 if model_valid else 0,
                f"{model_name}_statistical_reliability": self._calculate_model_reliability(validation_results)
            })

        logger.info(f"Model {model_name} logged with validation results")

    def _log_statistical_tests(self, test_results: Dict[str, StatisticalTest], category: str) -> None:
        """Log statistical test results to MLflow"""

        for test_name, test_result in test_results.items():
            # Log p-value
            mlflow.log_metric(f"pvalue_{category}_{test_name}", test_result.p_value)

            # Log test statistic
            mlflow.log_metric(f"statistic_{category}_{test_name}", test_result.statistic)

            # Log result (1 for pass, 0 for fail, 0.5 for warning)
            result_value = 1 if test_result.result == TestResult.PASS else (0.5 if test_result.result == TestResult.WARNING else 0)
            mlflow.log_metric(f"result_{category}_{test_name}", result_value)

            # Log critical values if available
            if test_result.critical_values:
                for cv_name, cv_value in test_result.critical_values.items():
                    mlflow.log_metric(f"critical_{category}_{test_name}_{cv_name}", cv_value)

    def _calculate_test_summary(self, test_results: Dict[str, StatisticalTest]) -> Dict[str, float]:
        """Calculate summary statistics for test results"""

        if not test_results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'warning': 0, 'pass_rate': 0}

        total = len(test_results)
        passed = sum(1 for result in test_results.values() if result.result == TestResult.PASS)
        failed = sum(1 for result in test_results.values() if result.result == TestResult.FAIL)
        warning = sum(1 for result in test_results.values() if result.result == TestResult.WARNING)
        pass_rate = passed / total if total > 0 else 0

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'warning': warning,
            'pass_rate': pass_rate
        }

    def _log_model_comparison_metrics(self, ml_results: Dict[str, StatisticalTest],
                                    models: Dict[str, Any]) -> None:
        """Log model comparison and ranking metrics"""

        # Extract permutation test results for ranking
        model_scores = {}
        for test_name, test_result in ml_results.items():
            if 'permutation' in test_name.lower():
                model_name = test_name.replace('_permutation', '')
                model_scores[model_name] = {
                    'mse': test_result.statistic,
                    'p_value': test_result.p_value,
                    'significant': test_result.result == TestResult.PASS
                }

        # Rank models by performance
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['mse'])

            for rank, (model_name, scores) in enumerate(sorted_models, 1):
                mlflow.log_metrics({
                    f"{model_name}_rank": rank,
                    f"{model_name}_mse": scores['mse'],
                    f"{model_name}_p_value": scores['p_value'],
                    f"{model_name}_is_significant": 1 if scores['significant'] else 0
                })

            # Log best model info
            if sorted_models:
                best_model_name, best_scores = sorted_models[0]
                mlflow.log_metrics({
                    "best_model_mse": best_scores['mse'],
                    "best_model_significant": 1 if best_scores['significant'] else 0
                })
                mlflow.log_param("best_model_name", best_model_name)

    def _calculate_reliability_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall reliability score for the analysis"""

        # Base score from test pass rate
        base_score = validation_report['summary']['tests_passed'] / max(1, validation_report['summary']['total_tests_performed'])

        # Penalties for critical failures
        penalties = 0

        # Check critical assumptions
        if not validation_report['statistical_assumptions'].get('var_model_valid', True):
            penalties += 0.2

        if not validation_report['statistical_assumptions'].get('spillover_significant', True):
            penalties += 0.1

        if not validation_report['statistical_assumptions'].get('ml_models_reliable', True):
            penalties += 0.2

        # Apply penalties
        reliability_score = max(0, base_score - penalties)

        return reliability_score

    def _calculate_model_reliability(self, validation_results: Dict[str, StatisticalTest]) -> float:
        """Calculate reliability score for a specific model"""

        relevant_tests = [
            result for result in validation_results.values()
            if any(keyword in result.name.lower() for keyword in ['permutation', 'cross', 'validation'])
        ]

        if not relevant_tests:
            return 0.0

        passed = sum(1 for result in relevant_tests if result.result == TestResult.PASS)
        return passed / len(relevant_tests)

    def _save_validation_artifact(self, data: Any, filename: str) -> None:
        """Save validation results as MLflow artifact"""

        # Convert StatisticalTest objects to serializable format
        if isinstance(data, dict):
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, StatisticalTest):
                    serializable_data[key] = {
                        'name': value.name,
                        'statistic': value.statistic,
                        'p_value': value.p_value,
                        'result': value.result.value,
                        'interpretation': value.interpretation,
                        'confidence_level': value.confidence_level,
                        'critical_values': value.critical_values
                    }
                else:
                    serializable_data[key] = value
        else:
            serializable_data = data

        # Save to temporary file and log as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(serializable_data, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, filename)

        # Clean up temporary file
        Path(temp_path).unlink()

    def _save_text_artifact(self, content: str, filename: str) -> None:
        """Save text content as MLflow artifact"""

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = f.name

        mlflow.log_artifact(temp_path, filename)
        Path(temp_path).unlink()

    def _generate_markdown_report(self, validation_report: Dict[str, Any]) -> str:
        """Generate comprehensive markdown validation report"""

        report_md = f"""
# Statistical Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Experiment:** {self.experiment_name}
**MLflow Run:** {self.current_run.info.run_id if self.current_run else 'N/A'}

## Executive Summary

- **Overall Validity:** {validation_report['summary']['overall_validity']}
- **Tests Performed:** {validation_report['summary']['total_tests_performed']}
- **Tests Passed:** {validation_report['summary']['tests_passed']} ({validation_report['summary']['tests_passed'] / max(1, validation_report['summary']['total_tests_performed']):.1%})
- **Tests Failed:** {validation_report['summary']['tests_failed']}
- **Reliability Score:** {self._calculate_reliability_score(validation_report):.3f}

## Statistical Assumptions Status

"""

        assumptions = validation_report.get('statistical_assumptions', {})
        for assumption, status in assumptions.items():
            status_emoji = "✅" if status else "❌"
            assumption_name = assumption.replace('_', ' ').title()
            report_md += f"- **{assumption_name}:** {status_emoji}\n"

        report_md += "\n## Recommendations\n\n"

        if validation_report.get('recommendations'):
            for rec in validation_report['recommendations']:
                report_md += f"- {rec}\n"
        else:
            report_md += "- No specific recommendations. Analysis appears statistically sound.\n"

        report_md += f"""

## Detailed Test Results

### VAR Model Validation
"""

        # Add detailed test results for each category
        for category_name, category_results in validation_report.get('detailed_results', {}).items():
            report_md += f"\n#### {category_name.replace('_', ' ').title()}\n\n"

            for test_name, test_result in category_results.items():
                if isinstance(test_result, dict) and 'name' in test_result:
                    # Serialized StatisticalTest
                    result_emoji = "✅" if test_result['result'] == 'PASS' else ("⚠️" if test_result['result'] == 'WARNING' else "❌")
                    report_md += f"- **{test_result['name']}** {result_emoji}\n"
                    report_md += f"  - Statistic: {test_result['statistic']:.4f}\n"
                    report_md += f"  - P-value: {test_result['p_value']:.4f}\n"
                    report_md += f"  - Interpretation: {test_result['interpretation']}\n\n"

        report_md += """

---

*This report was automatically generated by the Enhanced MLflow Statistical Validation Framework*
"""

        return report_md

    def end_run(self) -> None:
        """End current MLflow run with final validation summary"""

        if self.current_run:
            # Log final summary
            mlflow.set_tag("validation_completed", "true")
            mlflow.set_tag("end_time", datetime.now().isoformat())

            # Calculate and log overall metrics
            if self.validation_results:
                all_tests = []
                for category in self.validation_results.values():
                    all_tests.extend(category.values())

                if all_tests:
                    overall_pass_rate = sum(1 for test in all_tests if test.result == TestResult.PASS) / len(all_tests)
                    mlflow.log_metric("final_validation_score", overall_pass_rate)

            mlflow.end_run()
            self.current_run = None

            logger.info("MLflow run ended with validation summary")

    def __enter__(self):
        """Context manager entry"""
        if not self.current_run:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_run()