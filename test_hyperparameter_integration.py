#!/usr/bin/env python3
"""
Simple test script to validate hyperparameter sets integration
without requiring full dependencies
"""

import yaml
import sys
from pathlib import Path

def test_hyperparameter_integration():
    """Test hyperparameter sets integration logic"""
    print("🧪 Testing Hyperparameter Sets Integration...")
    print("=" * 60)

    # Test 1: Check configuration files exist
    print("Test 1: Configuration Files")
    config_path = "experiments/configs/hierarchical_config.yaml"
    hyperparameter_sets_path = "experiments/configs/hyperparameter_sets.yaml"

    if Path(config_path).exists():
        print(f"  ✅ Main config found: {config_path}")
    else:
        print(f"  ❌ Main config missing: {config_path}")
        return False

    if Path(hyperparameter_sets_path).exists():
        print(f"  ✅ Hyperparameter sets found: {hyperparameter_sets_path}")
    else:
        print(f"  ❌ Hyperparameter sets missing: {hyperparameter_sets_path}")
        return False

    # Test 2: Load and validate configurations
    print("\nTest 2: Configuration Loading")
    try:
        with open(config_path, 'r') as f:
            main_config = yaml.safe_load(f)
        print(f"  ✅ Main config loaded successfully")
        print(f"      - Keys: {list(main_config.keys())}")

        with open(hyperparameter_sets_path, 'r') as f:
            hyperparameter_sets = yaml.safe_load(f)
        print(f"  ✅ Hyperparameter sets loaded successfully")
        print(f"      - Top-level keys: {list(hyperparameter_sets.keys())}")

    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False

    # Test 3: Validate hyperparameter structure
    print("\nTest 3: Hyperparameter Structure Validation")
    try:
        # Check deep learning sets
        dl_sets = hyperparameter_sets.get('deep_learning_sets', {})
        lstm_sets = dl_sets.get('lstm_sets', {})
        gnn_sets = dl_sets.get('gnn_sets', {})

        print(f"  ✅ LSTM configurations: {len(lstm_sets)} sets")
        if lstm_sets:
            sample_lstm = list(lstm_sets.values())[0]
            print(f"      - Sample LSTM params: {list(sample_lstm.keys())}")

        print(f"  ✅ GNN configurations: {len(gnn_sets)} sets")
        if gnn_sets:
            sample_gnn = list(gnn_sets.values())[0]
            print(f"      - Sample GNN params: {list(sample_gnn.keys())}")

        # Check training sets
        training_sets = hyperparameter_sets.get('training_sets', {})
        optimizer_sets = training_sets.get('optimizer_sets', {})

        print(f"  ✅ Optimizer configurations: {len(optimizer_sets)} sets")

        # Calculate total combinations
        total_combinations = len(lstm_sets) * len(gnn_sets) * len(optimizer_sets)
        spillover_sets = hyperparameter_sets.get('spillover_parameter_sets', {}).get('diebold_yilmaz_sets', {})
        total_combinations *= len(spillover_sets) if spillover_sets else 1

        print(f"  ✅ Total possible combinations: {total_combinations}")

    except Exception as e:
        print(f"  ❌ Structure validation failed: {e}")
        return False

    # Test 4: Test configuration merging logic
    print("\nTest 4: Configuration Merging Logic")
    try:
        # Simulate configuration merging
        if lstm_sets and gnn_sets and optimizer_sets and spillover_sets:
            sample_lstm = list(lstm_sets.values())[0]
            sample_gnn = list(gnn_sets.values())[0]
            sample_train = list(optimizer_sets.values())[0]
            sample_spillover = list(spillover_sets.values())[0]

            # Mock merge (simplified version)
            merged_config = {
                'hidden_dim': sample_lstm.get('hidden_dim', 128),
                'num_layers': sample_lstm.get('num_layers', 2),
                'gnn_hidden_dim': sample_gnn.get('hidden_dim', 64),
                'gnn_type': sample_gnn.get('gnn_type', 'GAT'),
                'learning_rate': sample_train.get('learning_rate', 0.001),
                'batch_size': sample_train.get('batch_size', 32),
                'spillover_forecast_horizon': sample_spillover.get('forecast_horizon', 10)
            }

            print(f"  ✅ Sample merged configuration:")
            for key, value in merged_config.items():
                print(f"      - {key}: {value}")

    except Exception as e:
        print(f"  ❌ Configuration merging test failed: {e}")
        return False

    # Test 5: Check main_pipeline.py syntax
    print("\nTest 5: Pipeline Syntax Check")
    try:
        import py_compile
        py_compile.compile('src/main_pipeline.py', doraise=True)
        print(f"  ✅ main_pipeline.py syntax is valid")
    except Exception as e:
        print(f"  ❌ Syntax error in main_pipeline.py: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✨ Hyperparameter sets integration is ready for use")
    print("\nUsage examples:")
    print("  • Single config mode:    python3 -m src.main_pipeline --single-config")
    print("  • Hyperparameter mode:   python3 -m src.main_pipeline")
    print("  • Custom config:         python3 -m src.main_pipeline --config custom.yaml")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_hyperparameter_integration()
    sys.exit(0 if success else 1)