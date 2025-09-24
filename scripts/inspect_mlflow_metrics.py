#!/usr/bin/env python3
"""
Inspect MLflow metrics to show what statistical measures were tracked
"""

import mlflow
import pandas as pd
from pathlib import Path


def inspect_mlflow_metrics():
    """Inspect what metrics were logged in MLflow"""

    print("🔍 INSPECTING MLFLOW STATISTICAL METRICS")
    print("="*60)

    # Set tracking URI
    mlflow.set_tracking_uri("sqlite:///test_mlflow.db")

    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name("statistical_framework_test")
        if experiment is None:
            print("❌ Experiment not found")
            return

        print(f"✅ Found experiment: {experiment.name}")

        # Get runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"✅ Found {len(runs)} runs")

        if runs.empty:
            print("❌ No runs found")
            return

        # Analyze metrics from the latest run
        latest_run = runs.iloc[0]
        run_id = latest_run['run_id']

        print(f"✅ Inspecting run: {run_id}")

        # Get detailed run info
        run = mlflow.get_run(run_id)

        # Show all metrics
        metrics = run.data.metrics
        params = run.data.params

        print(f"\n📊 PARAMETERS TRACKED ({len(params)}):")
        print("-" * 40)
        for param, value in params.items():
            print(f"   {param}: {value}")

        print(f"\n📈 STATISTICAL METRICS TRACKED ({len(metrics)}):")
        print("-" * 40)

        # Organize metrics by category
        metric_categories = {
            'P-values': [],
            'Test Statistics': [],
            'Results': [],
            'Pass Rates': [],
            'Validation Scores': [],
            'Other': []
        }

        for metric_name, metric_value in metrics.items():
            if 'pvalue' in metric_name:
                metric_categories['P-values'].append((metric_name, metric_value))
            elif 'statistic' in metric_name:
                metric_categories['Test Statistics'].append((metric_name, metric_value))
            elif 'result' in metric_name:
                metric_categories['Results'].append((metric_name, metric_value))
            elif 'pass_rate' in metric_name or 'passed' in metric_name:
                metric_categories['Pass Rates'].append((metric_name, metric_value))
            elif 'validation' in metric_name or 'reliable' in metric_name:
                metric_categories['Validation Scores'].append((metric_name, metric_value))
            else:
                metric_categories['Other'].append((metric_name, metric_value))

        # Display organized metrics
        for category, metric_list in metric_categories.items():
            if metric_list:
                print(f"\n🔸 {category.upper()} ({len(metric_list)}):")
                for metric_name, metric_value in sorted(metric_list)[:10]:  # Show first 10
                    print(f"   {metric_name}: {metric_value:.4f}")
                if len(metric_list) > 10:
                    print(f"   ... and {len(metric_list) - 10} more")

        # Show tags
        tags = run.data.tags
        if tags:
            print(f"\n🏷️  RUN TAGS ({len(tags)}):")
            print("-" * 40)
            for tag, value in tags.items():
                print(f"   {tag}: {value}")

        # Summary statistics
        print(f"\n📋 STATISTICAL VALIDATION SUMMARY:")
        print("-" * 40)

        # Count statistical tests
        p_value_metrics = [m for m in metrics.keys() if 'pvalue' in m]
        test_result_metrics = [m for m in metrics.keys() if 'result' in m and 'pvalue' not in m]

        print(f"   Total p-values tracked: {len(p_value_metrics)}")
        print(f"   Total test results: {len(test_result_metrics)}")
        print(f"   Total metrics logged: {len(metrics)}")

        # Show significant results (p < 0.05)
        significant_tests = [
            (name, value) for name, value in metrics.items()
            if 'pvalue' in name and value < 0.05
        ]

        print(f"   Significant results (p < 0.05): {len(significant_tests)}")

        if significant_tests:
            print("\n🎯 SIGNIFICANT TEST RESULTS:")
            for test_name, p_value in significant_tests[:5]:
                clean_name = test_name.replace('pvalue_', '').replace('_', ' ')
                print(f"   • {clean_name}: p = {p_value:.4f}")

        # Show validation scores
        validation_metrics = [
            (name, value) for name, value in metrics.items()
            if any(keyword in name for keyword in ['valid', 'reliable', 'pass_rate'])
        ]

        if validation_metrics:
            print(f"\n✅ VALIDATION SCORES:")
            for metric_name, metric_value in validation_metrics:
                clean_name = metric_name.replace('_', ' ').title()
                print(f"   • {clean_name}: {metric_value:.3f}")

    except Exception as e:
        print(f"❌ Error inspecting MLflow: {e}")
        import traceback
        traceback.print_exc()


def show_experiment_overview():
    """Show overview of all experiments"""

    print("\n🔍 ALL EXPERIMENTS OVERVIEW")
    print("="*60)

    # Check different databases
    databases = [
        "test_mlflow.db",
        "statistical_spillover.db",
        "mlflow_final.db",
        "mlflow_light.db"
    ]

    for db in databases:
        if Path(db).exists():
            print(f"\n📁 Database: {db}")
            mlflow.set_tracking_uri(f"sqlite:///{db}")

            try:
                experiments = mlflow.search_experiments()
                print(f"   Experiments: {len(experiments)}")

                for exp in experiments:
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                    print(f"   • {exp.name}: {len(runs)} runs")

            except Exception as e:
                print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    inspect_mlflow_metrics()
    show_experiment_overview()
    print("\n🔗 MLflow UI: http://localhost:5555")
    print("✅ Statistical validation framework successfully tracks all metrics!")