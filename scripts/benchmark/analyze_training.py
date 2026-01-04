#!/usr/bin/env python
"""Training Benchmark Results Analyzer.

Analyzes completed benchmark runs by loading metrics DIRECTLY from run directories.
This ensures accurate analysis even when runs are re-executed after failures.

IMPORTANT: This script always scans runs/ directory to load fresh metrics.
It does NOT rely on benchmark_summary.json which may be stale.

Usage:
    # Analyze results from a benchmark run
    python scripts/benchmark/analyze_training.py \
        --input-dir artifacts/benchmark/main/20251216_123456/

    # Analyze with custom output directory
    python scripts/benchmark/analyze_training.py \
        --input-dir artifacts/benchmark/main/20251216_123456/ \
        --output-dir outputs/analysis/
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Execution order for consistent display (comprehensive list of all supported variants)
# Order: baseline methods first, then dueling methods, then rainbow methods
# Within each group: base -> cl -> per -> noisy -> per+noisy (each with and without cl)
EXECUTION_ORDER = [
    # Baseline variants
    "baseline",
    "baseline_cl",
    "baseline_per",
    "baseline_per_cl",
    "baseline_noisy",
    "baseline_noisy_cl",
    "baseline_per_noisy",
    "baseline_per_noisy_cl",
    # Dueling variants
    "dueling",
    "dueling_cl",
    "dueling_per",
    "dueling_per_cl",
    "dueling_noisy",
    "dueling_noisy_cl",
    "dueling_per_noisy",
    "dueling_per_noisy_cl",
    # Rainbow variants
    "rainbow",
    "rainbow_cl",
]


def load_metrics_from_run(run_dir: Path) -> Optional[dict]:
    """Load training metrics from a single run directory.

    Searches for metrics in order of preference:
    1. result.json at run directory level (summary from benchmark runner)
    2. Nested run subdirectory metrics files (e.g., run_*/metrics/training_metrics.json)
    3. Direct metrics files in run directory

    Args:
        run_dir: Path to run directory (e.g., runs/baseline_s42/)

    Returns:
        Dictionary with extracted metrics, or None if not found
    """
    # Parse variant and seed from directory name (e.g., "baseline_s42" -> variant="baseline", seed=42)
    dir_name = run_dir.name
    parts = dir_name.rsplit("_s", 1)
    if len(parts) != 2:
        print(f"  Warning: Could not parse variant/seed from directory name: {dir_name}")
        return None

    variant = parts[0]
    try:
        seed = int(parts[1])
    except ValueError:
        print(f"  Warning: Could not parse seed from: {parts[1]}")
        return None

    # Build result dict
    result = {
        "variant": variant,
        "seed": seed,
        "output_dir": str(run_dir),
        "metrics_source": None,
    }

    # PRIORITY 1: Check for result.json at run directory level
    # This is the summary file created by run_nips_benchmark.py
    result_file = run_dir / "result.json"
    if result_file.exists():
        try:
            with open(result_file) as f:
                result_data = json.load(f)

            # Use data from result.json
            result["status"] = result_data.get("status", "unknown")
            result["metrics_source"] = str(result_file)

            if result_data.get("best_reward") is not None:
                result["best_reward"] = float(result_data["best_reward"])
            if result_data.get("final_reward") is not None:
                result["final_reward"] = float(result_data["final_reward"])
            if result_data.get("total_episodes") is not None:
                result["total_episodes"] = result_data["total_episodes"]
            if result_data.get("error"):
                result["error"] = result_data["error"]

            # If we have best_reward from result.json, try to get detailed metrics
            # from nested run subdirectory for more analysis
            if result.get("best_reward") is not None:
                detailed_metrics = _load_detailed_metrics_from_nested(run_dir)
                if detailed_metrics:
                    result.update(detailed_metrics)
                    result["metrics_source"] = detailed_metrics.get("metrics_source", result["metrics_source"])
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not load {result_file}: {e}")

    # PRIORITY 2: Search nested run subdirectories (e.g., run_rainbow_drqn_*/metrics/)
    detailed_metrics = _load_detailed_metrics_from_nested(run_dir)
    if detailed_metrics and detailed_metrics.get("best_reward") is not None:
        result["status"] = "completed"
        result.update(detailed_metrics)
        return result

    # PRIORITY 3: Check direct metrics files in run directory
    metrics_paths = [
        run_dir / "metrics" / "training_metrics.json",
        run_dir / "metrics" / "final_metrics.json",
        run_dir / "training_metrics.json",
        run_dir / "metrics.json",
    ]

    for metrics_path in metrics_paths:
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics_data = json.load(f)
                result["metrics_source"] = str(metrics_path)
                result["status"] = "completed"
                _extract_rewards_from_metrics(result, metrics_data)
                return result
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {metrics_path}: {e}")
                continue

    # No metrics found - mark as failed
    result["status"] = "failed"
    result["error"] = "No metrics file found"
    return result


def _load_detailed_metrics_from_nested(run_dir: Path) -> Optional[dict]:
    """Load detailed metrics from nested run subdirectories.

    Searches for run_* subdirectories and loads metrics from them.

    Args:
        run_dir: Path to run directory

    Returns:
        Dictionary with detailed metrics, or None if not found
    """
    # Find nested run subdirectories (e.g., run_rainbow_drqn_20251216_140256/)
    nested_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    for nested_dir in nested_dirs:
        # Check for metrics in nested directory
        metrics_paths = [
            nested_dir / "metrics" / "training_metrics.json",
            nested_dir / "metrics" / "comprehensive" / "episodes_metrics.json",
            nested_dir / "metrics" / "final_metrics.json",
            nested_dir / "training_metrics.json",
        ]

        for metrics_path in metrics_paths:
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        metrics_data = json.load(f)

                    result = {"metrics_source": str(metrics_path)}
                    _extract_rewards_from_metrics(result, metrics_data)

                    if result.get("best_reward") is not None:
                        return result
                except (json.JSONDecodeError, IOError):
                    continue

    return None


def _extract_rewards_from_metrics(result: dict, metrics_data: dict) -> None:
    """Extract reward metrics from a metrics data dictionary.

    Modifies result dict in place to add extracted metrics.

    Args:
        result: Result dictionary to update
        metrics_data: Raw metrics data from JSON file
    """
    # Extract reward metrics - try multiple possible field names
    episode_rewards = (
        metrics_data.get("episode_rewards")
        or metrics_data.get("rewards")
        or metrics_data.get("test_rewards")
        or metrics_data.get("total_rewards")  # Common in training_metrics.json
        or []
    )

    if episode_rewards:
        result["best_reward"] = float(max(episode_rewards))
        result["final_reward"] = float(episode_rewards[-1])
        result["mean_reward"] = float(statistics.mean(episode_rewards))
        result["total_episodes"] = len(episode_rewards)

        # Compute stability from final portion of training (last 10%)
        final_start = max(0, int(len(episode_rewards) * 0.9))
        final_portion = episode_rewards[final_start:]
        if final_portion:
            result["final_mean_reward"] = float(statistics.mean(final_portion))
            if len(final_portion) > 1:
                result["final_std_reward"] = float(statistics.stdev(final_portion))
            else:
                result["final_std_reward"] = 0.0
    else:
        # Try to get from pre-computed fields
        if metrics_data.get("best_reward") is not None:
            result["best_reward"] = float(metrics_data["best_reward"])
        if metrics_data.get("final_reward") is not None:
            result["final_reward"] = float(metrics_data["final_reward"])
        if metrics_data.get("total_episodes") is not None:
            result["total_episodes"] = metrics_data["total_episodes"]
        if metrics_data.get("mean_reward") is not None:
            result["mean_reward"] = float(metrics_data["mean_reward"])

    # Get additional metrics if available
    if metrics_data.get("total_timesteps") is not None:
        result["total_timesteps"] = metrics_data["total_timesteps"]
    if metrics_data.get("training_time") is not None:
        result["training_time"] = metrics_data["training_time"]


def load_results_from_runs(input_dir: Path) -> List[dict]:
    """Load experiment results by scanning run directories.

    ALWAYS scans the runs/ subdirectory to get fresh metrics from each run.
    This ensures re-run experiments are properly captured.

    Args:
        input_dir: Directory containing benchmark results (with runs/ subdirectory)

    Returns:
        List of experiment result dictionaries
    """
    results = []

    runs_dir = input_dir / "runs"
    if not runs_dir.exists():
        print(f"Error: No runs directory found at {runs_dir}")
        return results

    # Scan all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if not run_dirs:
        print(f"Warning: No run directories found in {runs_dir}")
        return results

    print(f"Scanning {len(run_dirs)} run directories in {runs_dir}...")

    for run_dir in run_dirs:
        result = load_metrics_from_run(run_dir)
        if result:
            results.append(result)
            status = result.get("status", "unknown")
            reward = result.get("best_reward")
            reward_str = f"{reward:.4f}" if reward is not None else "N/A"
            print(f"  [{status.upper():>9}] {run_dir.name}: best_reward={reward_str}")

    return results


def compute_statistics(results: List[dict]) -> Dict:
    """Compute statistics for benchmark results.

    Args:
        results: List of experiment results

    Returns:
        Dictionary with computed statistics
    """
    completed = [
        r for r in results
        if r.get("status") == "completed" and r.get("best_reward") is not None
    ]

    # Group by variant
    variants = {}
    for r in completed:
        v = r["variant"]
        if v not in variants:
            variants[v] = []
        variants[v].append(r)

    stats = {
        "total_experiments": len(results),
        "completed": len(completed),
        "failed": len([r for r in results if r.get("status") in ("failed", "error", "timeout")]),
        "has_cl_variants": any("_cl" in r.get("variant", "") for r in results),
        "variants": {},
    }

    # Build ordered list of variants to process:
    # 1. First, include variants from EXECUTION_ORDER that exist in data
    # 2. Then, append any discovered variants not in EXECUTION_ORDER (sorted alphabetically)
    ordered_variants = [v for v in EXECUTION_ORDER if v in variants]
    extra_variants = sorted([v for v in variants if v not in EXECUTION_ORDER])
    if extra_variants:
        print(f"  Note: Found additional variants not in EXECUTION_ORDER: {extra_variants}")
    ordered_variants.extend(extra_variants)

    variant_stats = []
    for variant in ordered_variants:
        v_results = variants[variant]

        # Collect best_reward from each run (peak performance - what we'd checkpoint)
        best_rewards = [r.get("best_reward") for r in v_results if r.get("best_reward")]
        # Collect final_mean_reward (last 10% of training - stability metric)
        final_rewards = [r.get("final_mean_reward") or r.get("best_reward") for r in v_results]

        if best_rewards:
            mean_best = statistics.mean(best_rewards)
            std_best = statistics.stdev(best_rewards) if len(best_rewards) > 1 else 0.0
            mean_final = statistics.mean(final_rewards) if final_rewards else mean_best
            # Risk-adjusted score: Mean - 1*Std (penalizes variance for reliable deployment)
            risk_adjusted = mean_best - std_best

            variant_stat = {
                "variant": variant,
                "n_seeds": len(best_rewards),
                "mean_best_reward": mean_best,
                "mean_final_reward": mean_final,
                "std_reward": std_best,
                "risk_adjusted_score": risk_adjusted,  # Primary ranking metric
                "best_reward": max(best_rewards),
                "worst_reward": min(best_rewards),
                "stability": 1 - (std_best / mean_best) if mean_best > 0 else 0,
                "rewards": best_rewards,
                "seeds": [r["seed"] for r in v_results],
            }
            stats["variants"][variant] = variant_stat
            # Rank by risk-adjusted score (balances performance and reliability)
            variant_stats.append((variant, risk_adjusted))

    # Compute rankings
    variant_stats.sort(key=lambda x: x[1], reverse=True)
    for rank, (variant, _) in enumerate(variant_stats, 1):
        if variant in stats["variants"]:
            stats["variants"][variant]["rank"] = rank

    # Compute interaction effects if factorial design
    if stats["has_cl_variants"]:
        stats["interaction_effects"] = {}
        stats["missing_interaction_pairs"] = {}  # Track incomplete pairs
        # Check all base methods for CL interaction effects
        for base in ["baseline", "dueling", "rainbow"]:
            has_base = base in stats["variants"]
            has_cl = f"{base}_cl" in stats["variants"]

            if has_base and has_cl:
                # Both variants exist - compute interaction effect
                no_cl = stats["variants"][base]["mean_best_reward"]
                with_cl = stats["variants"][f"{base}_cl"]["mean_best_reward"]
                effect = with_cl - no_cl
                stats["interaction_effects"][base] = {
                    "no_cl": no_cl,
                    "with_cl": with_cl,
                    "effect": effect,
                }
            elif has_base or has_cl:
                # Only one variant exists - record as incomplete pair
                stats["missing_interaction_pairs"][base] = {
                    "has_base": has_base,
                    "has_cl": has_cl,
                    "missing": f"{base}" if not has_base else f"{base}_cl",
                }

    return stats


def generate_summary_report(
    results: List[dict], output_dir: Path, stats: Optional[Dict] = None
) -> str:
    """Generate a markdown summary report.

    Args:
        results: List of experiment results
        output_dir: Output directory
        stats: Pre-computed statistics (optional)

    Returns:
        Markdown report content
    """
    if stats is None:
        stats = compute_statistics(results)

    # Determine benchmark type
    has_cl_variants = stats["has_cl_variants"]
    if has_cl_variants:
        benchmark_type = "2x2 Factorial (Method x Curriculum Learning)"
    else:
        benchmark_type = "Rainbow DRQN vs Baseline DRQN (No Curriculum Learning)"

    report = f"""# NIPS Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Output Directory:** `{output_dir}`
**Benchmark Type:** {benchmark_type}
**Analysis Method:** Direct metrics scan from run directories

## Summary

| Status | Count |
|--------|-------|
| Completed | {stats['completed']} |
| Failed | {stats['failed']} |
| **Total** | **{stats['total_experiments']}** |

## Results Comparison

*Ranked by Risk-Adjusted Score = Mean Best - Std (higher is better). This balances peak performance with reliability across seeds. Methods with high variance are penalized.*

"""
    variant_stats = stats["variants"]

    if len(variant_stats) >= 2:
        if has_cl_variants:
            report += "| Method | CL | Seeds | Mean Best | Std | Risk-Adj | Rank |\n"
            report += "|--------|-----|-------|-----------|-----|----------|------|\n"
        else:
            report += "| Method | Seeds | Mean Best | Std | Risk-Adj | Winner |\n"
            report += "|--------|-------|-----------|-----|----------|--------|\n"

        # Sort by rank
        sorted_variants = sorted(
            variant_stats.items(),
            key=lambda x: x[1].get("rank", 999)
        )

        for variant, v_stats in sorted_variants:
            rank = v_stats.get("rank", "N/A")
            if has_cl_variants:
                # Build descriptive method name with all components
                if "rainbow" in variant:
                    method_name = "Rainbow DRQN"
                elif "dueling" in variant:
                    method_name = "Dueling DRQN"
                else:
                    method_name = "Baseline DRQN"
                # Add component flags to method name
                components = []
                if "_per" in variant:
                    components.append("PER")
                if "_noisy" in variant:
                    components.append("Noisy")
                if components:
                    method_name += " + " + " + ".join(components)
                cl_status = "Yes" if "_cl" in variant else "No"
                rank_marker = "**1st**" if rank == 1 else f"{rank}th"
                bold = "**" if rank == 1 else ""
                mean_best = v_stats.get('mean_best_reward', v_stats.get('mean_reward', 0))
                risk_adj = v_stats.get('risk_adjusted_score', mean_best - v_stats['std_reward'])
                report += f"| {bold}{method_name}{bold} | {bold}{cl_status}{bold} | {v_stats['n_seeds']} | {bold}{mean_best:.4f}{bold} | {v_stats['std_reward']:.4f} | {bold}{risk_adj:.4f}{bold} | {rank_marker} |\n"
            else:
                winner = "**Winner**" if rank == 1 else ""
                bold = "**" if rank == 1 else ""
                mean_best = v_stats.get('mean_best_reward', v_stats.get('mean_reward', 0))
                risk_adj = v_stats.get('risk_adjusted_score', mean_best - v_stats['std_reward'])
                report += f"| {bold}{variant}{bold} | {v_stats['n_seeds']} | {bold}{mean_best:.4f}{bold} | {v_stats['std_reward']:.4f} | {bold}{risk_adj:.4f}{bold} | {winner} |\n"

    report += "\n## Key Findings\n\n"

    if has_cl_variants and "interaction_effects" in stats:
        report += "### Method x Curriculum Interaction\n\n"
        report += "*Mean Best Reward comparison for base methods with/without Curriculum Learning. CL Effect = With CL - No CL.*\n\n"
        report += "| Method | No CL | With CL | CL Effect |\n"
        report += "|--------|-------|---------|----------|\n"
        for base, effect_data in stats["interaction_effects"].items():
            effect = effect_data["effect"]
            effect_str = f"+{effect:.4f}" if effect > 0 else f"{effect:.4f}"
            report += f"| {base.title()} | {effect_data['no_cl']:.4f} | {effect_data['with_cl']:.4f} | {effect_str} |\n"

        # Show incomplete pairs (missing variants)
        if stats.get("missing_interaction_pairs"):
            report += "\n**Incomplete Method Pairs** (cannot compute CL effect):\n\n"
            report += "| Method | Has Base | Has CL | Missing Variant |\n"
            report += "|--------|----------|--------|----------------|\n"
            for base, pair_data in stats["missing_interaction_pairs"].items():
                has_base = "✓" if pair_data["has_base"] else "✗"
                has_cl = "✓" if pair_data["has_cl"] else "✗"
                report += f"| {base.title()} | {has_base} | {has_cl} | `{pair_data['missing']}` |\n"
            report += "\n*To compute CL effect, run the benchmark with the missing variant(s).*\n"

        report += "\n"

    # Winner summary
    if variant_stats:
        best_variant = min(variant_stats.items(), key=lambda x: x[1].get("rank", 999))
        risk_adj = best_variant[1].get('risk_adjusted_score', 0)
        mean_best = best_variant[1].get('mean_best_reward', 0)
        std = best_variant[1].get('std_reward', 0)
        report += f"**Winner:** {best_variant[0]} (Risk-Adj: {risk_adj:.4f} = {mean_best:.4f} - {std:.4f})\n\n"

    report += "## Paper Integration\n\n"
    report += "Copy `tables/training_comparison.tex` to `nips_paper/second_draft/tables/`\n\n"

    report += "## Individual Results\n\n"

    # Sort results by variant and seed for organized display
    sorted_results = sorted(results, key=lambda r: (r.get("variant", ""), r.get("seed", 0)))

    for r in sorted_results:
        status = r.get("status", "unknown")
        status_marker = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "error": "[ERR]",
            "timeout": "[TIMEOUT]",
        }.get(status, "[?]")

        report += f"### {status_marker} {r.get('variant', 'unknown')} (seed={r.get('seed', '?')})\n\n"
        report += f"- **Status:** {status}\n"
        report += f"- **Output:** `{r.get('output_dir', 'N/A')}`\n"

        if r.get("metrics_source"):
            report += f"- **Metrics Source:** `{r['metrics_source']}`\n"
        if r.get("best_reward") is not None:
            report += f"- **Best Reward:** {r['best_reward']:.4f}\n"
        if r.get("final_mean_reward") is not None:
            report += f"- **Final Mean Reward (last 10%):** {r['final_mean_reward']:.4f}\n"
        if r.get("final_reward") is not None:
            report += f"- **Final Reward:** {r['final_reward']:.4f}\n"
        if r.get("total_episodes"):
            report += f"- **Total Episodes:** {r['total_episodes']}\n"
        if r.get("error"):
            report += f"- **Error:** {r['error'][:200]}\n"

        report += "\n"

    return report


def analyze_results(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Main analysis function.

    Scans run directories to load metrics directly from each experiment.
    Does NOT use benchmark_summary.json to ensure fresh analysis.

    Args:
        input_dir: Directory containing benchmark results (with runs/ subdirectory)
        output_dir: Output directory for analysis results

    Returns:
        Dictionary with analysis results
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Always load from run directories for fresh metrics
    print(f"\n{'='*60}")
    print("NIPS BENCHMARK ANALYSIS")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")

    results = load_results_from_runs(input_dir)

    if not results:
        print("\nNo results found to analyze")
        return {}

    # Set output directory
    if output_dir is None:
        output_dir = input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {output_dir}")

    # Compute statistics
    stats = compute_statistics(results)

    # Generate report
    report_content = generate_summary_report(results, input_dir, stats)

    # Save outputs
    report_file = output_dir / "BENCHMARK_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"\n[OK] Report: {report_file}")

    stats_file = output_dir / "analysis_results.json"
    with open(stats_file, "w") as f:
        # Convert to JSON-serializable format (exclude raw rewards list)
        json_stats = {k: v for k, v in stats.items() if k != "variants"}
        json_stats["variants"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "rewards"}
            for k, v in stats.get("variants", {}).items()
        }
        json.dump(json_stats, f, indent=2)
    print(f"[OK] Statistics: {stats_file}")

    # Generate CSV for easy import
    csv_file = output_dir / "method_comparison.csv"
    with open(csv_file, "w") as f:
        # Use comprehensive CSV format with all component flags
        f.write("variant,base_method,per,noisy,curriculum,n_seeds,mean_best,std,risk_adjusted,mean_final,rank\n")
        for variant, v_stats in stats["variants"].items():
            # Parse variant components
            if "rainbow" in variant:
                base_method = "rainbow"
            elif "dueling" in variant:
                base_method = "dueling"
            else:
                base_method = "baseline"
            per_flag = "yes" if "_per" in variant else "no"
            noisy_flag = "yes" if "_noisy" in variant else "no"
            cl_flag = "yes" if "_cl" in variant else "no"
            mean_best = v_stats.get('mean_best_reward', v_stats.get('mean_reward', 0))
            mean_final = v_stats.get('mean_final_reward', mean_best)
            risk_adj = v_stats.get('risk_adjusted_score', mean_best - v_stats['std_reward'])
            f.write(
                f"{variant},{base_method},{per_flag},{noisy_flag},{cl_flag},"
                f"{v_stats['n_seeds']},{mean_best:.4f},{v_stats['std_reward']:.4f},{risk_adj:.4f},{mean_final:.4f},{v_stats.get('rank', 'N/A')}\n"
            )

    # Also generate a simplified CSV for backward compatibility
    csv_simple_file = output_dir / "method_comparison_simple.csv"
    with open(csv_simple_file, "w") as f:
        if stats["has_cl_variants"]:
            f.write("method,curriculum,n_seeds,mean_best,std,risk_adjusted,rank\n")
            for variant, v_stats in stats["variants"].items():
                # Determine base method for simplified view
                if "rainbow" in variant:
                    base_method = "rainbow_drqn"
                elif "dueling" in variant:
                    base_method = "dueling_drqn"
                else:
                    base_method = "baseline_drqn"
                cl_status = "enabled" if "_cl" in variant else "disabled"
                mean_best = v_stats.get('mean_best_reward', v_stats.get('mean_reward', 0))
                risk_adj = v_stats.get('risk_adjusted_score', mean_best - v_stats['std_reward'])
                f.write(
                    f"{base_method},{cl_status},{v_stats['n_seeds']},{mean_best:.4f},"
                    f"{v_stats['std_reward']:.4f},{risk_adj:.4f},{v_stats.get('rank', 'N/A')}\n"
                )
        else:
            f.write("method,n_seeds,mean_best,std,risk_adjusted,rank\n")
            for variant, v_stats in stats["variants"].items():
                mean_best = v_stats.get('mean_best_reward', v_stats.get('mean_reward', 0))
                risk_adj = v_stats.get('risk_adjusted_score', mean_best - v_stats['std_reward'])
                f.write(
                    f"{variant},{v_stats['n_seeds']},{mean_best:.4f},"
                    f"{v_stats['std_reward']:.4f},{risk_adj:.4f},{v_stats.get('rank', 'N/A')}\n"
                )
    print(f"[OK] CSV: {csv_file}")
    print(f"[OK] CSV (simplified): {csv_simple_file}")

    # Save raw results for reference
    raw_results_file = output_dir / "raw_results.json"
    with open(raw_results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Raw results: {raw_results_file}")

    return {
        "stats": stats,
        "results": results,
        "report_file": str(report_file),
        "stats_file": str(stats_file),
        "csv_file": str(csv_file),
        "csv_simple_file": str(csv_simple_file),
        "raw_results_file": str(raw_results_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NIPS benchmark results from run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script ALWAYS scans the runs/ subdirectory to load metrics directly
from each experiment's output files. This ensures that re-run experiments
are properly captured in the analysis.

Examples:
    # Analyze from benchmark directory
    python scripts/nips_benchmark/analyze_results.py \\
        --input-dir artifacts/benchmark/nips_benchmark/20251216_123456/

    # Custom output directory
    python scripts/nips_benchmark/analyze_results.py \\
        --input-dir artifacts/benchmark/nips_benchmark/20251216_123456/ \\
        --output-dir outputs/custom_analysis/
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results (must have runs/ subdirectory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: input_dir/analysis/)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    try:
        result = analyze_results(input_dir, output_dir)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

        if result:
            stats = result["stats"]
            print(f"Experiments: {stats['completed']}/{stats['total_experiments']} completed")
            if stats["variants"]:
                best = min(stats["variants"].items(), key=lambda x: x[1].get("rank", 999))
                print(f"Winner: {best[0]}")
                risk_adj = best[1].get('risk_adjusted_score', 0)
                mean_best = best[1].get('mean_best_reward', 0)
                std = best[1].get('std_reward', 0)
                print(f"  Risk-Adjusted Score: {risk_adj:.4f} (= {mean_best:.4f} - {std:.4f})")
            print(f"\nFull report: {result['report_file']}")

        print(f"{'='*60}")

    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
