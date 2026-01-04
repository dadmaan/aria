#!/usr/bin/env python
"""Curriculum Learning Ablation Runner.

Executes benchmark experiments for Curriculum Learning ablation study:
- No Curriculum (flat action space)
- Two Stage Curriculum (2-phase progressive learning)
- Three Stage Curriculum (3-phase progressive learning)

Each configuration is run with multiple seeds for statistical significance.
Supports parallel execution for faster benchmarking.

Usage:
    # Run all variants with default seeds
    python scripts/benchmark/run_curriculum_ablation.py

    # Run specific variants
    python scripts/benchmark/run_curriculum_ablation.py \
        --variants no_curriculum two_stage three_stage \
        --seeds 42 123 456

    # Run in parallel (4 workers)
    python scripts/benchmark/run_curriculum_ablation.py --parallel --workers 4

    # Dry run to see commands
    python scripts/benchmark/run_curriculum_ablation.py --dry-run
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os


# Configuration file mapping
VARIANT_CONFIGS = {
    "no_curriculum": "configs/ablations/curriculum/no_curriculum_flat.yaml",
    "two_stage": "configs/ablations/curriculum/two_stage_curriculum.yaml",
    "three_stage": "configs/ablations/curriculum/three_stage_curriculum.yaml",
}

# Execution order (simplest first)
EXECUTION_ORDER = [
    "no_curriculum",
    "two_stage",
    "three_stage",
]


def run_experiment(
    variant: str,
    seed: int,
    output_dir: Path,
    config_path: str,
    dry_run: bool = False,
    no_wandb: bool = False,
) -> dict:
    """Run a single training experiment.

    Args:
        variant: Variant name (e.g., 'baseline', 'cl_enabled')
        seed: Random seed
        output_dir: Base output directory
        config_path: Path to configuration file
        dry_run: If True, print command without executing
        no_wandb: If True, disable WandB logging

    Returns:
        dict with experiment result
    """
    exp_dir = output_dir / variant / f"seed_{seed}"
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

    if no_wandb:
        cmd.append("--no-wandb")

    result = {
        "variant": variant,
        "seed": seed,
        "config": config_path,
        "output_dir": str(exp_dir),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "final_reward": None,
        "best_reward": None,
        "total_episodes": None,
        "phase_transitions": None,
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"Running: {variant} (seed={seed})")
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
            timeout=3600,  # 1 hour timeout
        )

        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "completed"

            # Try to extract metrics from output
            output_lines = process.stdout.split("\n")
            for line in output_lines:
                line_lower = line.lower()
                if "best_reward" in line_lower or "best reward" in line_lower:
                    try:
                        import re

                        numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                        if numbers:
                            result["best_reward"] = float(numbers[-1])
                    except:
                        pass
                elif "final" in line_lower and "reward" in line_lower:
                    try:
                        import re

                        numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                        if numbers:
                            result["final_reward"] = float(numbers[-1])
                    except:
                        pass
                elif "phase" in line_lower and "transition" in line_lower:
                    try:
                        import re

                        numbers = re.findall(r"\d+", line)
                        if numbers:
                            result["phase_transitions"] = int(numbers[-1])
                    except:
                        pass

            # Try to load metrics from saved file
            metrics_file = exp_dir / "metrics" / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                result["total_episodes"] = metrics.get("total_episodes")
                if "episode_rewards" in metrics:
                    rewards = metrics["episode_rewards"]
                    if rewards:
                        result["final_reward"] = float(rewards[-1])
                        result["best_reward"] = float(max(rewards))

            print(
                f"✅ {variant} (seed={seed}) completed - Best: {result.get('best_reward', 'N/A')}"
            )
        else:
            result["status"] = "failed"
            result["error"] = (
                process.stderr[-500:] if process.stderr else "Unknown error"
            )
            print(f"❌ {variant} (seed={seed}) failed: {result['error'][:200]}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment exceeded 1 hour timeout"
        result["end_time"] = datetime.now().isoformat()
        print(f"⏰ {variant} (seed={seed}) timed out")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"❌ {variant} (seed={seed}) error: {e}")

    # Save result to JSON
    result_file = exp_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_experiment_wrapper(args: Tuple) -> dict:
    """Wrapper for parallel execution.

    Args:
        args: Tuple of (variant, seed, output_dir, config_path, dry_run, no_wandb)

    Returns:
        Experiment result dict
    """
    variant, seed, output_dir, config_path, dry_run, no_wandb = args
    return run_experiment(variant, seed, output_dir, config_path, dry_run, no_wandb)


def run_parallel(
    experiments: List[Tuple],
    workers: int,
) -> List[dict]:
    """Run experiments in parallel.

    Args:
        experiments: List of (variant, seed, output_dir, config_path, dry_run) tuples
        workers: Number of parallel workers

    Returns:
        List of experiment results
    """
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment_wrapper, exp): exp for exp in experiments
        }

        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                variant, seed, _, _, _ = exp
                results.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                    }
                )
                print(f"❌ {variant} (seed={seed}) parallel execution error: {e}")

    return results


def generate_summary_report(results: List[dict], output_dir: Path) -> str:
    """Generate a markdown summary report.

    Args:
        results: List of experiment results
        output_dir: Output directory

    Returns:
        Markdown report content
    """
    completed = [r for r in results if r["status"] == "completed"]

    report = f"""# Curriculum Learning Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory:** `{output_dir}`

## Summary

| Status | Count |
|--------|-------|
| Completed | {len([r for r in results if r['status'] == 'completed'])} |
| Failed | {len([r for r in results if r['status'] == 'failed'])} |
| Error | {len([r for r in results if r['status'] == 'error'])} |
| Timeout | {len([r for r in results if r['status'] == 'timeout'])} |
| **Total** | **{len(results)}** |

## Results by Variant

"""
    # Group by variant
    variants = {}
    for r in completed:
        v = r["variant"]
        if v not in variants:
            variants[v] = []
        variants[v].append(r)

    if variants:
        report += "| Variant | Seeds | Mean Reward | Std | Best | Phases |\n"
        report += "|---------|-------|-------------|-----|------|--------|\n"

        for variant in EXECUTION_ORDER:
            if variant in variants:
                v_results = variants[variant]
                rewards = [r["best_reward"] for r in v_results if r.get("best_reward")]
                phases = [
                    r["phase_transitions"]
                    for r in v_results
                    if r.get("phase_transitions")
                ]

                if rewards:
                    import statistics

                    mean_r = statistics.mean(rewards)
                    std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0
                    best_r = max(rewards)
                    phases_str = str(phases[0]) if phases else "N/A"
                    report += f"| {variant} | {len(v_results)} | {mean_r:.4f} | {std_r:.4f} | {best_r:.4f} | {phases_str} |\n"

    report += "\n## Individual Results\n\n"

    for r in results:
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "error": "⚠️",
            "timeout": "⏰",
            "dry_run": "🔍",
        }.get(r["status"], "❓")

        report += f"### {status_emoji} {r['variant']} (seed={r['seed']})\n\n"
        report += f"- **Status:** {r['status']}\n"
        report += f"- **Config:** `{r['config']}`\n"
        report += f"- **Output:** `{r['output_dir']}`\n"

        if r.get("best_reward"):
            report += f"- **Best Reward:** {r['best_reward']:.4f}\n"
        if r.get("final_reward"):
            report += f"- **Final Reward:** {r['final_reward']:.4f}\n"
        if r.get("phase_transitions"):
            report += f"- **Phase Transitions:** {r['phase_transitions']}\n"
        if r.get("error"):
            report += f"- **Error:** {r['error'][:200]}\n"

        report += "\n"

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run Curriculum Learning benchmark variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all variants with default seeds
    python scripts/benchmark/run_curriculum_ablation.py

    # Run specific variants
    python scripts/benchmark/run_curriculum_ablation.py \\
        --variants no_curriculum two_stage three_stage

    # Run with custom seeds
    python scripts/benchmark/run_curriculum_ablation.py \\
        --seeds 42 123 456 789 1000

    # Run in parallel (faster but uses more resources)
    python scripts/benchmark/run_curriculum_ablation.py --parallel --workers 4

    # Dry run to see commands without executing
    python scripts/benchmark/run_curriculum_ablation.py --dry-run

    # Disable WandB logging
    python scripts/benchmark/run_curriculum_ablation.py --no-wandb
        """,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["no_curriculum", "three_stage"],
        choices=list(VARIANT_CONFIGS.keys()),
        help="CL variants to benchmark (default: no_curriculum three_stage)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for experiments (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/cl_benchmark_TIMESTAMP)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging for all experiments",
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"artifacts/benchmark/{datetime.now().strftime('%Y%m%d_%H%M%S')}_cl_variants"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create missing configs if requested or needed
    base_dir = Path(__file__).parent.parent.parent

    # Verify configs exist
    missing_configs = []
    for variant in args.variants:
        config_path = base_dir / VARIANT_CONFIGS[variant]
        if not config_path.exists():
            missing_configs.append(variant)

    if missing_configs:
        print(f"⚠️  Missing config files for: {missing_configs}")
        print("Run with --create-configs to generate them, or use available variants:")
        available = [v for v in args.variants if v not in missing_configs]
        print(f"  Available: {available}")

    # Sort variants by execution order
    variants = [v for v in EXECUTION_ORDER if v in args.variants]

    print(f"\n{'='*60}")
    print("Curriculum Learning Benchmark Campaign")
    print(f"{'='*60}")
    print(f"Variants: {variants}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print(f"Parallel: {args.parallel} (workers={args.workers})")
    print(f"Total experiments: {len(variants) * len(args.seeds)}")
    print(f"{'='*60}\n")

    # Build experiment list
    experiments = []
    for variant in variants:
        config_path = VARIANT_CONFIGS[variant]
        for seed in args.seeds:
            experiments.append(
                (variant, seed, output_dir, config_path, args.dry_run, args.no_wandb)
            )

    # Run experiments
    if args.parallel and not args.dry_run:
        print(
            f"🚀 Running {len(experiments)} experiments in parallel ({args.workers} workers)..."
        )
        results = run_parallel(experiments, args.workers)
    else:
        results = []
        for exp in experiments:
            variant, seed, out_dir, config_path, dry_run, no_wandb = exp
            result = run_experiment(
                variant, seed, out_dir, config_path, dry_run, no_wandb
            )
            results.append(result)

            # Check for failure
            if result["status"] in ("failed", "error") and not args.continue_on_error:
                print(
                    f"\n❌ Stopping due to failure. Use --continue-on-error to proceed."
                )
                break

    # Summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] in ("failed", "error", "timeout")]

    print(f"Completed: {len(completed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if completed:
        print("\nResults by Variant:")
        for variant in variants:
            variant_results = [r for r in completed if r["variant"] == variant]
            if variant_results:
                rewards = [
                    r["best_reward"] for r in variant_results if r.get("best_reward")
                ]
                if rewards:
                    import statistics

                    mean_reward = statistics.mean(rewards)
                    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0
                    print(
                        f"  {variant}: {mean_reward:.4f} ± {std_reward:.4f} (n={len(rewards)})"
                    )

    # Save summary JSON
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "variants": variants,
                "seeds": args.seeds,
                "parallel": args.parallel,
                "workers": args.workers,
                "results": results,
                "completed": len(completed),
                "failed": len(failed),
            },
            f,
            indent=2,
        )
    print(f"\nJSON summary: {summary_file}")

    # Generate markdown report
    report_content = generate_summary_report(results, output_dir)
    report_file = output_dir / "BENCHMARK_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"Markdown report: {report_file}")


if __name__ == "__main__":
    main()
