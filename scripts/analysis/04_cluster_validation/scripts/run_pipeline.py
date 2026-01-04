#!/usr/bin/env python3
"""
GHSOM Listening Test Pipeline - Main Orchestration Script

This script coordinates the entire workflow for preparing GHSOM cluster samples
for listening tests: selection, audio rendering, and similarity analysis.

Author: GHSOM Analysis Pipeline
Date: 2025-11-19
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


class ListeningTestPipeline:
    """Orchestrate the complete listening test preparation workflow."""

    def __init__(self, config_file: Path):
        """
        Initialize pipeline with configuration.

        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.results = {
            "start_time": datetime.now().isoformat(),
            "config_file": str(config_file),
            "steps_completed": [],
            "steps_failed": [],
        }

    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        with open(self.config_file, "r") as f:
            config = json.load(f)

        # Convert string paths to Path objects
        for key in ["model_dir", "metadata_csv", "midi_root", "output_root"]:
            if key in config:
                config[key] = Path(config[key])

        return config

    def _run_step(self, step_name: str, command: list) -> bool:
        """
        Execute a pipeline step.

        Args:
            step_name: Name of the step
            command: Command list to execute

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*70}")
        print(f"STEP: {step_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(str(c) for c in command)}\n")

        try:
            result = subprocess.run(
                command, check=True, capture_output=False, text=True
            )

            self.results["steps_completed"].append(
                {
                    "step": step_name,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(f"\n✓ {step_name} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.results["steps_failed"].append(
                {
                    "step": step_name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(f"\n✗ {step_name} failed: {e}")
            return False

    def step_1_sample_selection(self) -> bool:
        """Step 1: Select representative samples from clusters."""
        config = self.config["sampling"]
        model_dir = self.config["model_dir"]

        output_dir = self.config["output_root"] / "sampled_midi"

        command = [
            sys.executable,
            str(Path(__file__).parent / "01_select_cluster_samples.py"),
            "--cluster-csv",
            str(model_dir / "sample_to_cluster.csv"),
            "--metadata-csv",
            str(self.config["metadata_csv"]),
            "--midi-root",
            str(self.config["midi_root"]),
            "--output-dir",
            str(output_dir),
            "--samples-per-cluster",
            str(config["samples_per_cluster"]),
            "--sampling-method",
            config["sampling_method"],
            "--random-seed",
            str(config["random_seed"]),
        ]

        return self._run_step("Sample Selection", command)

    def step_2_audio_rendering(self) -> bool:
        """Step 2: Render MIDI files to audio."""
        config = self.config["rendering"]

        input_dir = self.config["output_root"] / "sampled_midi"
        output_dir = self.config["output_root"] / "rendered_audio"

        command = [
            sys.executable,
            str(Path(__file__).parent / "02_render_audio.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--sample-rate",
            str(config["sample_rate"]),
            "--format",
            config["format"],
            "--parallel-jobs",
            str(config["parallel_jobs"]),
        ]

        if config.get("soundfont"):
            command.extend(["--soundfont", str(config["soundfont"])])

        if config.get("no_normalize", False):
            command.append("--no-normalize")

        return self._run_step("Audio Rendering", command)

    def step_3_similarity_analysis(self) -> bool:
        """Step 3: Analyze intra-cluster similarity."""
        config = self.config["analysis"]
        model_dir = self.config["model_dir"]

        output_dir = self.config["output_root"] / "similarity_analysis"

        command = [
            sys.executable,
            str(Path(__file__).parent / "03_analyze_cluster_similarity.py"),
            "--cluster-csv",
            str(model_dir / "sample_to_cluster.csv"),
            "--metadata-csv",
            str(self.config["metadata_csv"]),
            "--output-dir",
            str(output_dir),
            "--distance-metric",
            config["distance_metric"],
        ]

        return self._run_step("Similarity Analysis", command)

    def save_results(self) -> None:
        """Save pipeline execution results."""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["success"] = len(self.results["steps_failed"]) == 0

        results_file = self.config["output_root"] / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'='*70}")
        print("PIPELINE RESULTS")
        print(f"{'='*70}")
        print(f"Steps completed: {len(self.results['steps_completed'])}")
        print(f"Steps failed: {len(self.results['steps_failed'])}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*70}\n")

    def run(self, skip_steps: list = None) -> bool:
        """
        Execute the complete pipeline.

        Args:
            skip_steps: List of step numbers to skip (e.g., [1, 2])

        Returns:
            True if all steps succeeded, False otherwise
        """
        skip_steps = skip_steps or []

        print(f"\n{'='*70}")
        print("GHSOM LISTENING TEST PIPELINE")
        print(f"{'='*70}")
        print(f"Configuration: {self.config_file}")
        print(f"Output root: {self.config['output_root']}")
        print(f"{'='*70}\n")

        # Create output directory
        self.config["output_root"].mkdir(parents=True, exist_ok=True)

        # Execute steps
        steps = [
            (1, self.step_1_sample_selection),
            (2, self.step_2_audio_rendering),
            (3, self.step_3_similarity_analysis),
        ]

        for step_num, step_func in steps:
            if step_num in skip_steps:
                print(f"\nSkipping step {step_num}: {step_func.__name__}")
                continue

            success = step_func()
            if not success and self.config.get("stop_on_error", True):
                print(f"\n✗ Pipeline stopped due to error in step {step_num}")
                self.save_results()
                return False

        self.save_results()

        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        print(f"Output directory: {self.config['output_root']}")
        print(f"  - sampled_midi/         : Selected MIDI files")
        print(f"  - rendered_audio/       : Rendered audio files")
        print(f"  - similarity_analysis/  : Analysis results")
        print(f"  - pipeline_results.json : Execution log\n")

        return True


def create_default_config(model_dir: Path, output_file: Path) -> None:
    """Create a default configuration file template."""
    config = {
        "model_dir": str(model_dir),
        "metadata_csv": str(
            Path(
                "/workspace/artifacts/features/raw/commu_full/features_with_metadata.csv"
            )
        ),
        "midi_root": str(Path("/workspace/data/raw/commu/full")),
        "output_root": str(Path("/workspace/outputs/ghsom_cluster_analysis")),
        "sampling": {
            "samples_per_cluster": 5,
            "sampling_method": "random",
            "random_seed": 42,
            "min_samples_threshold": 3,
        },
        "rendering": {
            "sample_rate": 44100,
            "format": "wav",
            "normalize": True,
            "parallel_jobs": 4,
            "soundfont": None,
        },
        "analysis": {"distance_metric": "euclidean"},
        "stop_on_error": True,
    }

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created default configuration at: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="GHSOM Listening Test Pipeline - Orchestrate sampling, rendering, and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --config config.json
  
  # Create default configuration
  python run_pipeline.py --create-config --model-dir /path/to/model
  
  # Skip audio rendering step
  python run_pipeline.py --config config.json --skip 2
        """,
    )

    parser.add_argument("--config", type=Path, help="Path to configuration JSON file")
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file",
    )
    parser.add_argument(
        "--model-dir", type=Path, help="Model directory (required with --create-config)"
    )
    parser.add_argument(
        "--skip", type=int, nargs="+", help="Step numbers to skip (e.g., --skip 2 3)"
    )

    args = parser.parse_args()

    # Create config mode
    if args.create_config:
        if not args.model_dir:
            print("Error: --model-dir required with --create-config")
            return 1

        output_file = args.model_dir / "pipeline_config.json"
        create_default_config(args.model_dir, output_file)
        return 0

    # Run pipeline mode
    if not args.config:
        print("Error: --config required (or use --create-config)")
        parser.print_help()
        return 1

    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    # Execute pipeline
    pipeline = ListeningTestPipeline(args.config)
    success = pipeline.run(skip_steps=args.skip or [])

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
