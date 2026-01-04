#!/usr/bin/env python
"""Algorithm Ablation Runner.

Executes benchmark experiments for Rainbow DQN algorithm ablation study:
- Baseline (DQN)
- Dueling DQN
- C51 (Distributional)
- Rainbow (Full combination)

Each configuration is run with multiple seeds for statistical significance.

Usage:
    python scripts/benchmark/run_algorithm_ablation.py \
        --algorithms baseline dueling_dqn c51 rainbow \
        --seeds 42 123 456 789 \
        --output-dir outputs/algorithm_ablation
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


# Configuration file mapping
ALGORITHM_CONFIGS = {
    "baseline": "configs/ablations/algorithm/baseline.yaml",
    "dueling_dqn": "configs/ablations/algorithm/dueling_drqn.yaml",
    "c51": "configs/ablations/algorithm/c51.yaml",
    "rainbow": "configs/ablations/algorithm/rainbow_drqn.yaml",
}

# Risk-stratified execution order
EXECUTION_ORDER = ["baseline", "dueling_dqn", "c51", "rainbow"]


def run_experiment(
    algorithm: str,
    seed: int,
    output_dir: Path,
    config_path: str,
    dry_run: bool = False,
) -> dict:
    """Run a single training experiment.

    Args:
        algorithm: Algorithm name
        seed: Random seed
        output_dir: Base output directory
        config_path: Path to configuration file
        dry_run: If True, print command without executing

    Returns:
        dict with experiment result
    """
    exp_dir = output_dir / algorithm / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/training/run_training.py",
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--output-dir",
        str(exp_dir),
    ]

    result = {
        "algorithm": algorithm,
        "seed": seed,
        "config": config_path,
        "output_dir": str(exp_dir),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "final_reward": None,
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"Running: {algorithm} (seed={seed})")
    print(f"Config: {config_path}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}\n")

    result["start_time"] = datetime.now().isoformat()

    try:
        # Run training
        process = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
        )

        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "completed"
            # Try to extract final reward from output
            output_lines = process.stdout.split("\n")
            for line in reversed(output_lines):
                if "best_reward" in line.lower() or "final reward" in line.lower():
                    try:
                        # Extract number from line
                        import re

                        numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                        if numbers:
                            result["final_reward"] = float(numbers[-1])
                            break
                    except:
                        pass
            print(f"✅ {algorithm} (seed={seed}) completed")
        else:
            result["status"] = "failed"
            result["error"] = (
                process.stderr[-500:] if process.stderr else "Unknown error"
            )
            print(f"❌ {algorithm} (seed={seed}) failed: {result['error'][:200]}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"❌ {algorithm} (seed={seed}) error: {e}")

    # Save result to JSON
    result_file = exp_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Rainbow DQN benchmark")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=EXECUTION_ORDER,
        choices=list(ALGORITHM_CONFIGS.keys()),
        help="Algorithms to benchmark",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789],
        help="Random seeds for experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/rainbow_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for results",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with other experiments if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort algorithms by execution order
    algorithms = [a for a in EXECUTION_ORDER if a in args.algorithms]

    print(f"\n{'='*60}")
    print("Rainbow DQN Benchmark Campaign")
    print(f"{'='*60}")
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print(f"Total experiments: {len(algorithms) * len(args.seeds)}")
    print(f"{'='*60}\n")

    # Run experiments
    results = []

    for algo in algorithms:
        config_path = ALGORITHM_CONFIGS[algo]

        for seed in args.seeds:
            result = run_experiment(
                algorithm=algo,
                seed=seed,
                output_dir=output_dir,
                config_path=config_path,
                dry_run=args.dry_run,
            )
            results.append(result)

            # Check for failure
            if result["status"] == "failed" and not args.continue_on_error:
                print(
                    f"\n❌ Stopping due to failure. Use --continue-on-error to proceed."
                )
                break

        if results[-1]["status"] == "failed" and not args.continue_on_error:
            break

    # Summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] in ("failed", "error")]

    print(f"Completed: {len(completed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if completed:
        print("\nResults:")
        for algo in algorithms:
            algo_results = [r for r in completed if r["algorithm"] == algo]
            if algo_results:
                rewards = [r["final_reward"] for r in algo_results if r["final_reward"]]
                if rewards:
                    import statistics

                    mean_reward = statistics.mean(rewards)
                    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0
                    print(f"  {algo}: {mean_reward:.4f} ± {std_reward:.4f}")

    # Save summary
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "algorithms": algorithms,
                "seeds": args.seeds,
                "results": results,
                "completed": len(completed),
                "failed": len(failed),
            },
            f,
            indent=2,
        )

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
