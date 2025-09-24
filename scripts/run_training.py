#!/usr/bin/env python3
"""
Script to run the comprehensive training pipeline with optimized hyperparameters
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/Hudini/projects/info_spillover')

from src.training_pipeline import ComprehensiveTrainingPipeline


def main():
    """Run the comprehensive training pipeline"""

    print("🚀 STARTING COMPREHENSIVE TRAINING PIPELINE")
    print("="*80)

    # Path to optimized hyperparameters configuration
    config_path = "/home/Hudini/projects/info_spillover/experiments/configs/optimized_hyperparameters.yaml"

    try:
        # Initialize and run pipeline
        pipeline = ComprehensiveTrainingPipeline(config_path)
        pipeline.run_comprehensive_pipeline()

        print("\n🎉 Training pipeline completed successfully!")
        print("🔗 Check MLflow UI for detailed results")

    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()