#!/usr/bin/env python3
"""
Comprehensive Feature Analysis Pipeline

Orchestrates multiple feature analysis methods:
1. Exploratory Data Analysis (EDA)
2. Recursive Feature Elimination (RFE)
3. SHAP Analysis
4. Consensus Ranking

Usage:
    python run_comprehensive_feature_analysis.py --data_path <path> --output_dir <dir>
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for logging imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from utils.logging.logging_manager import LoggingManager


class FeatureAnalysisPipeline:
    """Orchestrates comprehensive feature analysis pipeline."""

    def __init__(self, data_path: str, output_dir: str, logger: LoggingManager):
        """Initialize pipeline.

        Args:
            data_path: Path to input feature CSV
            output_dir: Output directory for results
            logger: LoggingManager instance
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Validate inputs
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define script paths
        scripts_dir = Path(__file__).parent
        self.scripts = {
            "eda": scripts_dir / "feature_distributions.py",
            "rfe": scripts_dir / "rfe_analysis.py",
            "shap": scripts_dir / "shap_analysis.py",
            "consensus": scripts_dir.parent / "consensus_analysis.py",
        }

        # Validate scripts exist
        for name, path in self.scripts.items():
            if not path.exists():
                raise FileNotFoundError(f"Script not found: {path}")

        self.results = {}
        self.stage_times = {}

    def run_stage(
        self, stage_name: str, script_path: Path, extra_args: list = None
    ) -> bool:
        """Run a pipeline stage.

        Args:
            stage_name: Name of the stage
            script_path: Path to Python script
            extra_args: Additional command-line arguments

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE: %s", stage_name.upper())
        self.logger.info("=" * 80)

        start_time = time.time()

        # Build command with absolute paths
        abs_data_path = self.data_path.resolve()
        abs_output_dir = self.output_dir.resolve()

        cmd = [
            sys.executable,
            str(script_path),
            "--data_path",
            str(abs_data_path),
            "--output_dir",
            str(abs_output_dir),
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            # Run script from workspace root
            workspace_root = Path(__file__).resolve().parent.parent.parent.parent
            result = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                check=True,
            )

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    self.logger.info(line)

            elapsed = time.time() - start_time
            self.stage_times[stage_name] = elapsed

            self.logger.info("✓ Stage completed in %.2f seconds", elapsed)
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error("✗ Stage failed with exit code %d", e.returncode)
            if e.stdout:
                self.logger.error("STDOUT:\n%s", e.stdout)
            if e.stderr:
                self.logger.error("STDERR:\n%s", e.stderr)

            elapsed = time.time() - start_time
            self.stage_times[stage_name] = elapsed

            return False

    def run_pipeline(self) -> bool:
        """Execute full pipeline.

        Returns:
            True if all stages successful, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE FEATURE ANALYSIS PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info("Data: %s", self.data_path)
        self.logger.info("Output: %s", self.output_dir)
        self.logger.info("=" * 80)

        pipeline_start = time.time()

        # Stage 1: EDA
        if not self.run_stage("eda", self.scripts["eda"]):
            self.logger.error("Pipeline failed at EDA stage")
            return False

        # Stage 2: RFE
        if not self.run_stage("rfe", self.scripts["rfe"]):
            self.logger.error("Pipeline failed at RFE stage")
            return False

        # Stage 3: SHAP
        if not self.run_stage("shap", self.scripts["shap"]):
            self.logger.error("Pipeline failed at SHAP stage")
            return False

        # Stage 4: Consensus
        if not self.run_stage("consensus", self.scripts["consensus"]):
            self.logger.error("Pipeline failed at Consensus stage")
            return False

        # Pipeline complete
        total_time = time.time() - pipeline_start

        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info("Total execution time: %.2f seconds", total_time)
        self.logger.info("")
        self.logger.info("Stage timings:")
        for stage, elapsed in self.stage_times.items():
            self.logger.info("  %s: %.2f seconds", stage, elapsed)

        # Save pipeline summary
        summary = {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "total_time_seconds": total_time,
            "stage_times": self.stage_times,
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Pipeline summary saved to: %s", summary_path)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive feature importance analysis pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="artifacts/features/raw/commu_full/features_numeric.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory path (default: outputs/feature_importance_analysis/comprehensive_{dataset_name}_{timestamp})",
    )

    args = parser.parse_args()

    # Extract dataset name from data path
    data_path = Path(args.data_path)
    dataset_name = data_path.parent.name  # Gets 'commu_full' from path

    # Set default output directory with dataset name if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/feature_importance_analysis/comprehensive_{dataset_name}_{timestamp}"

    # Initialize logging
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "pipeline.log"

    logger = LoggingManager(
        name="feature_analysis_pipeline",
        log_file=str(log_file),
        enable_wandb=False,
    )

    try:
        # Create and run pipeline
        pipeline = FeatureAnalysisPipeline(args.data_path, args.output_dir, logger)
        success = pipeline.run_pipeline()

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error("Pipeline failed with exception: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
