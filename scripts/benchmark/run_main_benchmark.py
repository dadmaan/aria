#!/usr/bin/env python
"""Main Benchmark Runner.

Runs controlled experiments comparing Rainbow DRQN vs Baseline DRQN
in a 2×2 factorial design (Method × Curriculum Learning).

This benchmark provides complete transparency by testing ALL four conditions:
1. Baseline DRQN - No CL (control)
2. Baseline DRQN - With CL
3. Rainbow DRQN - No CL (expected best)
4. Rainbow DRQN - With CL (negative control - expected worst)

Usage:
    # Quick run (3 seeds, NO CL only) - Recommended for time-constrained experiments
    python scripts/benchmark/run_main_benchmark.py --quick

    # Full run (5 seeds, NO CL only) - For statistical rigor
    python scripts/benchmark/run_main_benchmark.py --full

    # Full 2×2 factorial (all 4 conditions) - Complete transparency
    python scripts/benchmark/run_main_benchmark.py --full --include-cl

    # CL variants only
    python scripts/benchmark/run_main_benchmark.py --full --variants baseline_cl rainbow_cl

    # Dry run to see commands
    python scripts/benchmark/run_main_benchmark.py --dry-run --include-cl

    # Skip analysis/table generation
    python scripts/benchmark/run_main_benchmark.py --quick --skip-analysis --skip-tables

Expected Results (based on prior analysis):
    - Rainbow DRQN (No CL):   0.653 ± 0.001, Stability: 0.953 (BEST)
    - Baseline DRQN (No CL):  0.643 ± 0.005, Stability: 0.940
    - Baseline DRQN (CL):     ~0.62 ± 0.02
    - Rainbow DRQN (CL):      0.571 ± 0.085 (WORST - CL interferes with PER)
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Import analysis and table generation modules
from analyze_training import analyze_results, generate_summary_report, compute_statistics
from export_latex_tables import generate_tables, generate_latex_table


# NIPS Benchmark Configuration - Full Factorial Design
VARIANT_CONFIGS = {
    # === Original 2×2 Factorial (Method × CL) ===
    # No Curriculum Learning
    "baseline": "configs/benchmark/main/baseline_no_cl.yaml",
    "rainbow": "configs/benchmark/main/rainbow_no_cl.yaml",
    # With Curriculum Learning
    "baseline_cl": "configs/benchmark/main/baseline_cl.yaml",
    "rainbow_cl": "configs/benchmark/main/rainbow_cl.yaml",

    # === Ablation: Dueling Only ===
    "dueling": "configs/benchmark/main/dueling_no_cl.yaml",
    "dueling_cl": "configs/benchmark/main/dueling_cl.yaml",

    # === Ablation: Baseline + PER ===
    "baseline_per": "configs/benchmark/main/baseline_per_no_cl.yaml",
    "baseline_per_cl": "configs/benchmark/main/baseline_per_cl.yaml",

    # === Ablation: Baseline + NoisyNet ===
    "baseline_noisy": "configs/benchmark/main/baseline_noisy_no_cl.yaml",
    "baseline_noisy_cl": "configs/benchmark/main/baseline_noisy_cl.yaml",

    # === Ablation: Baseline + PER + NoisyNet (Rainbow w/o C51) ===
    "baseline_per_noisy": "configs/benchmark/main/baseline_per_noisy_no_cl.yaml",
    "baseline_per_noisy_cl": "configs/benchmark/main/baseline_per_noisy_cl.yaml",

    # === Ablation: Dueling + PER ===
    "dueling_per": "configs/benchmark/main/dueling_per_no_cl.yaml",
    "dueling_per_cl": "configs/benchmark/main/dueling_per_cl.yaml",

    # === Ablation: Dueling + NoisyNet ===
    "dueling_noisy": "configs/benchmark/main/dueling_noisy_no_cl.yaml",
    "dueling_noisy_cl": "configs/benchmark/main/dueling_noisy_cl.yaml",

    # === Ablation: Dueling + PER + NoisyNet (Rainbow w/o C51) ===
    "dueling_per_noisy": "configs/benchmark/main/dueling_per_noisy_no_cl.yaml",
    "dueling_per_noisy_cl": "configs/benchmark/main/dueling_per_noisy_cl.yaml",
}

# Default variants (original 2×2 factorial)
DEFAULT_VARIANTS = ["baseline", "rainbow"]

# Original 2×2 factorial variants
ORIGINAL_VARIANTS = ["baseline", "baseline_cl", "rainbow", "rainbow_cl"]

# Dueling ablation variants
DUELING_VARIANTS = ["dueling", "dueling_cl"]

# PER ablation variants (baseline and dueling)
PER_VARIANTS = ["baseline_per", "baseline_per_cl", "dueling_per", "dueling_per_cl"]

# NoisyNet ablation variants (baseline and dueling)
NOISY_VARIANTS = ["baseline_noisy", "baseline_noisy_cl", "dueling_noisy", "dueling_noisy_cl"]

# PER+NoisyNet ablation variants (Rainbow without C51)
PER_NOISY_VARIANTS = ["baseline_per_noisy", "baseline_per_noisy_cl", "dueling_per_noisy", "dueling_per_noisy_cl"]

# CL-only ablation variants (efficient design - skip no-CL since CL works for simple methods)
CL_ONLY_ABLATION = [
    "dueling_cl",
    "baseline_per_cl", "baseline_noisy_cl", "baseline_per_noisy_cl",
    "dueling_per_cl", "dueling_noisy_cl", "dueling_per_noisy_cl",
]

# All variants for full ablation study
ALL_VARIANTS = list(VARIANT_CONFIGS.keys())

# Execution order (controls first, then treatments)
EXECUTION_ORDER = [
    # Original
    "baseline", "baseline_cl", "rainbow", "rainbow_cl",
    # Dueling
    "dueling", "dueling_cl",
    # PER ablation
    "baseline_per", "baseline_per_cl", "dueling_per", "dueling_per_cl",
    # NoisyNet ablation
    "baseline_noisy", "baseline_noisy_cl", "dueling_noisy", "dueling_noisy_cl",
    # PER+NoisyNet ablation
    "baseline_per_noisy", "baseline_per_noisy_cl", "dueling_per_noisy", "dueling_per_noisy_cl",
]

# Seed configurations
QUICK_SEEDS = [42, 123, 456]
FULL_SEEDS = [42, 123, 456, 789, 202]


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
        variant: Variant name ('baseline' or 'rainbow')
        seed: Random seed
        output_dir: Base output directory
        config_path: Path to configuration file
        dry_run: If True, print command without executing
        no_wandb: If True, disable WandB logging

    Returns:
        dict with experiment result
    """
    exp_dir = output_dir / "runs" / f"{variant}_s{seed}"
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
        "--wandb-tags",
        f"nips_benchmark,{variant},seed{seed}",
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
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"NIPS BENCHMARK: {variant.upper()} (seed={seed})")
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
            timeout=7200,  # 2 hour timeout (increased for full training)
        )

        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "completed"

            # Try to load metrics from saved file (primary source - most reliable)
            metrics_file = exp_dir / "metrics" / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                result["total_episodes"] = metrics.get("total_episodes")
                if "episode_rewards" in metrics:
                    rewards = metrics["episode_rewards"]
                    if rewards:
                        # Use mean of last 100 episodes for final_reward (more stable)
                        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
                        result["final_reward"] = float(sum(last_100) / len(last_100))
                        result["best_reward"] = float(max(rewards))

            # Fallback: Try to extract metrics from stdout (less reliable)
            if result["best_reward"] is None or result["final_reward"] is None:
                output_lines = process.stdout.split("\n")
                for line in output_lines:
                    line_lower = line.lower()
                    # Only parse lines that look like metric reports, not file paths
                    # Avoid lines with paths like "artifacts/benchmark/nips_benchmark/20251216_140256"
                    if "artifact" in line_lower or "output" in line_lower or "/" in line:
                        continue
                    if result["best_reward"] is None and ("best_reward" in line_lower or "best reward" in line_lower):
                        try:
                            import re
                            # Match decimal numbers only (to avoid timestamps)
                            numbers = re.findall(r"[-+]?\d+\.\d+", line)
                            if numbers:
                                result["best_reward"] = float(numbers[-1])
                        except:
                            pass
                    elif result["final_reward"] is None and "final" in line_lower and "reward" in line_lower:
                        try:
                            import re
                            # Match decimal numbers only (to avoid timestamps)
                            numbers = re.findall(r"[-+]?\d+\.\d+", line)
                            if numbers:
                                result["final_reward"] = float(numbers[-1])
                        except:
                            pass

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
        result["error"] = "Experiment exceeded 2 hour timeout"
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
    """Wrapper for parallel execution."""
    variant, seed, output_dir, config_path, dry_run, no_wandb = args
    return run_experiment(variant, seed, output_dir, config_path, dry_run, no_wandb)


def run_parallel(experiments: List[Tuple], workers: int) -> List[dict]:
    """Run experiments in parallel."""
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
                variant, seed, _, _, _, _ = exp
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


def main():
    parser = argparse.ArgumentParser(
        description="Run NIPS Benchmark: 2×2 Factorial (Method × Curriculum Learning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected Results (based on prior analysis):
    - Rainbow DRQN (No CL):   0.653 ± 0.001, Stability: 0.953 (BEST)
    - Baseline DRQN (No CL):  0.643 ± 0.005, Stability: 0.940
    - Baseline DRQN (CL):     ~0.62 ± 0.02
    - Rainbow DRQN (CL):      0.571 ± 0.085 (WORST)

Examples:
    # Quick run (3 seeds, NO CL only) - Recommended for experiments
    python scripts/benchmark/run_main_benchmark.py --quick

    # Full run (5 seeds, NO CL only) - For statistical rigor
    python scripts/benchmark/run_main_benchmark.py --full

    # Full 2×2 factorial (all 4 conditions) - Complete transparency
    python scripts/benchmark/run_main_benchmark.py --full --include-cl

    # CL variants only
    python scripts/benchmark/run_main_benchmark.py --full --variants baseline_cl rainbow_cl

    # Dry run to see commands
    python scripts/benchmark/run_main_benchmark.py --dry-run --include-cl
        """,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with 3 seeds (42, 123, 456) - Recommended for paper",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run with 5 seeds for statistical rigor",
    )
    parser.add_argument(
        "--include-cl",
        action="store_true",
        help="Include curriculum learning variants (full 2×2 factorial design)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["dueling", "per", "noisy", "per_noisy", "all"],
        default=None,
        help="Run specific ablation group: dueling, per, noisy, per_noisy, or all",
    )
    parser.add_argument(
        "--cl-only",
        action="store_true",
        help="Run CL-only ablation (7 variants: dueling_cl + all component_cl variants). "
             "Efficient design since we know CL works for simple methods.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Custom random seeds",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(VARIANT_CONFIGS.keys()),
        help="Methods to benchmark (default: baseline rainbow; with --include-cl: all 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/benchmark/nips_benchmark_TIMESTAMP)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (use --workers to set count)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "-j", "--parallel-runs",
        type=int,
        default=None,
        metavar="N",
        help="Run N experiments in parallel (shorthand for --parallel --workers N)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip running analysis after experiments complete",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip generating LaTeX tables after experiments complete",
    )
    args = parser.parse_args()

    # Handle -j/--parallel-runs shorthand
    if args.parallel_runs is not None:
        args.parallel = True
        args.workers = args.parallel_runs

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.full:
        seeds = FULL_SEEDS
    elif args.quick:
        seeds = QUICK_SEEDS
    else:
        seeds = [42]  # Default: single seed

    # Determine variants
    if args.variants:
        variants_to_run = args.variants
    elif args.cl_only:
        # Efficient CL-only ablation (7 variants)
        variants_to_run = CL_ONLY_ABLATION
    elif args.ablation:
        # Select ablation group
        if args.ablation == "dueling":
            variants_to_run = DUELING_VARIANTS
        elif args.ablation == "per":
            variants_to_run = PER_VARIANTS
        elif args.ablation == "noisy":
            variants_to_run = NOISY_VARIANTS
        elif args.ablation == "per_noisy":
            variants_to_run = PER_NOISY_VARIANTS
        elif args.ablation == "all":
            variants_to_run = ALL_VARIANTS
        else:
            variants_to_run = DEFAULT_VARIANTS
    elif args.include_cl:
        variants_to_run = ORIGINAL_VARIANTS
    else:
        variants_to_run = DEFAULT_VARIANTS

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cl_suffix = "_full_factorial" if args.include_cl else ""
        args.output_dir = f"artifacts/benchmark/nips_benchmark/{timestamp}{cl_suffix}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify configs exist
    base_dir = Path(__file__).parent.parent.parent
    for variant in variants_to_run:
        config_path = base_dir / VARIANT_CONFIGS[variant]
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            print("Please run benchmark setup first.")
            sys.exit(1)

    variants = [v for v in EXECUTION_ORDER if v in variants_to_run]

    # Determine benchmark type for display
    if set(variants) == set(ALL_VARIANTS):
        benchmark_type = "Full 2×2 Factorial (Method × Curriculum)"
    elif args.include_cl:
        benchmark_type = "Partial Factorial (selected variants)"
    else:
        benchmark_type = "No CL (Rainbow vs Baseline)"

    print(f"\n{'='*60}")
    print("NIPS BENCHMARK: Rainbow DRQN vs Baseline DRQN")
    print(f"{'='*60}")
    print(f"Design: {benchmark_type}")
    print(f"Methods: {variants}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")
    print(f"Parallel: {args.parallel} (workers={args.workers})")
    print(f"Total experiments: {len(variants) * len(seeds)}")
    print(f"Expected time: ~{len(variants) * len(seeds) * 20} minutes (GPU)")
    print(f"{'='*60}\n")

    # Build experiment list
    experiments = []
    for variant in variants:
        config_path = VARIANT_CONFIGS[variant]
        for seed in seeds:
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

    # Summary
    print(f"\n{'='*60}")
    print("NIPS BENCHMARK SUMMARY")
    print(f"{'='*60}")

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] in ("failed", "error", "timeout")]

    print(f"Completed: {len(completed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if completed:
        print("\nResults by Method:")
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
                "benchmark_type": "nips_rainbow_vs_baseline",
                "variants": variants,
                "seeds": seeds,
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

    # Generate analysis report (using external script)
    report_file = None
    if not args.skip_analysis:
        print("\n📊 Running analysis...")
        try:
            analysis_output = output_dir / "analysis"
            analysis_result = analyze_results(
                input_dir=output_dir,
                output_dir=analysis_output,
            )
            report_file = analysis_result.get("report_file")
            print(f"Analysis report: {report_file}")
        except Exception as e:
            print(f"⚠️ Analysis failed: {e}")
            # Fallback to simple inline report
            report_content = generate_summary_report(results, output_dir)
            report_file = output_dir / "BENCHMARK_REPORT.md"
            with open(report_file, "w") as f:
                f.write(report_content)
            print(f"Fallback report: {report_file}")
    else:
        print("\n⏭️ Skipping analysis (--skip-analysis)")

    # Generate LaTeX tables (using external script)
    tables_dir = output_dir / "tables"
    if not args.skip_tables and completed:
        print("\n📄 Generating LaTeX tables...")
        try:
            table_files = generate_tables(
                input_dir=output_dir,
                output_dir=tables_dir,
            )
            if table_files:
                print(f"Tables saved to: {tables_dir}")
        except Exception as e:
            print(f"⚠️ Table generation failed: {e}")
            # Fallback to inline generation
            try:
                generate_latex_table(results, tables_dir)
            except Exception as e2:
                print(f"⚠️ Fallback table generation also failed: {e2}")
    elif args.skip_tables:
        print("\n⏭️ Skipping table generation (--skip-tables)")

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    if report_file:
        print(f"1. Review results: {report_file}")
    else:
        print(f"1. Review results: {summary_file}")
    if not args.skip_tables:
        print(
            f"2. Copy LaTeX table: cp {tables_dir}/training_comparison.tex nips_paper/second_draft/tables/"
        )
        print(f"3. Update 06_experiments.tex with new table")
    else:
        print("2. Generate tables: python scripts/benchmark/export_latex_tables.py --input-dir {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
