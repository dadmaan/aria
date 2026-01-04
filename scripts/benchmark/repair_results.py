#!/usr/bin/env python3
"""Repair Result Files.

This script fixes missing or buggy result.json files by reading the actual
training metrics and generating correct summary files.

Usage:
    python scripts/benchmark/repair_results.py <benchmark_dir>

Example:
    python scripts/benchmark/repair_results.py \
        artifacts/benchmark/main/20251216_140256
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def find_training_run_dir(run_dir: Path) -> Optional[Path]:
    """Find the actual training run subdirectory.

    The structure is: runs/<variant_seed>/run_*_<timestamp>/
    """
    for subdir in run_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("run_"):
            return subdir
    return None


def load_training_metrics(training_run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load training metrics from the run directory."""
    metrics_file = training_run_dir / "metrics" / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def load_config(training_run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load config.yaml from the run directory."""
    import yaml
    config_file = training_run_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return None


def extract_variant_and_seed(run_name: str) -> tuple:
    """Extract variant and seed from run directory name.

    Examples:
        baseline_s42 -> ('baseline', 42)
        rainbow_cl_s123 -> ('rainbow_cl', 123)
    """
    parts = run_name.rsplit('_s', 1)
    if len(parts) == 2:
        variant = parts[0]
        try:
            seed = int(parts[1])
            return variant, seed
        except ValueError:
            pass
    return run_name, None


def generate_result_json(run_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Generate result.json for a single run directory.

    Args:
        run_dir: Path to the run directory (e.g., runs/baseline_cl_s42/)
        force: If True, regenerate even if result.json exists

    Returns:
        The generated result dictionary
    """
    run_name = run_dir.name
    result_file = run_dir / "result.json"

    # Check if result.json already exists and is valid
    if result_file.exists() and not force:
        with open(result_file) as f:
            existing = json.load(f)
        # Check if the existing file has correct values (not timestamps)
        final_reward = existing.get("final_reward")
        if final_reward is not None and abs(final_reward) < 10:  # Valid reward range
            print(f"  [SKIP] {run_name}: result.json already exists with valid data")
            return existing

    # Find training run subdirectory
    training_run_dir = find_training_run_dir(run_dir)
    if not training_run_dir:
        print(f"  [ERROR] {run_name}: No training run directory found")
        return {"error": "No training run directory found"}

    # Load training metrics
    metrics = load_training_metrics(training_run_dir)
    if not metrics:
        print(f"  [ERROR] {run_name}: No training_metrics.json found")
        return {"error": "No training_metrics.json found"}

    # Load config
    config = load_config(training_run_dir)

    # Extract variant and seed
    variant, seed = extract_variant_and_seed(run_name)

    # Calculate metrics
    episode_rewards = metrics.get("episode_rewards", [])
    total_episodes = metrics.get("total_episodes", len(episode_rewards))

    if episode_rewards:
        # Use mean of last 100 episodes for final_reward (more stable)
        last_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        final_reward = sum(last_100) / len(last_100)
        best_reward = max(episode_rewards)
    else:
        final_reward = None
        best_reward = None

    # Determine config path
    config_path = None
    if config:
        # Try to find relative config path
        config_path = config.get("experiment_name", f"configs/nips_benchmark/{variant}.yaml")

    # Build result dictionary
    result = {
        "variant": variant,
        "seed": seed,
        "config": config_path,
        "output_dir": str(training_run_dir),
        "status": "completed",
        "start_time": None,  # Would need to parse from logs
        "end_time": None,
        "final_reward": final_reward,
        "best_reward": best_reward,
        "total_episodes": total_episodes,
        "error": None,
        "_regenerated": True,
        "_regenerated_at": datetime.now().isoformat(),
    }

    # Try to get timing info from training_summary.json
    summary_file = training_run_dir / "metrics" / "comprehensive" / "training_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        if "summary/total_training_time_sec" in summary:
            # Estimate end_time based on training time
            training_time = summary["summary/total_training_time_sec"]
            result["training_time_sec"] = training_time

    # Save result.json
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    status = "REGENERATED" if result_file.exists() else "CREATED"
    print(f"  [{status}] {run_name}: final_reward={final_reward:.4f}, best_reward={best_reward:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate result.json files from training data"
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Path to benchmark directory (e.g., artifacts/benchmark/nips_benchmark/20251216_140256)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all result.json files, even if they exist"
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Only regenerate for a specific run (e.g., baseline_cl_s42)"
    )

    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return 1

    runs_dir = benchmark_dir / "runs"
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1

    print(f"Regenerating result.json files in: {benchmark_dir}")
    print("=" * 60)

    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Filter by specific run if requested
        if args.run and run_dir.name != args.run:
            continue

        result = generate_result_json(run_dir, force=args.force)
        results.append(result)

    print("=" * 60)
    print(f"Processed {len(results)} runs")

    # Summary
    successful = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if "error" in r and r["error"])
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    return 0


if __name__ == "__main__":
    exit(main())
