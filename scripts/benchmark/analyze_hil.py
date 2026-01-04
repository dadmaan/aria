#!/usr/bin/env python
"""HIL (Human-in-the-Loop) Inference Results Analyzer.

Analyzes HIL preference adaptation results to understand how models perform
in deployment scenarios, not just training benchmarks.

Key metrics analyzed:
- Feedback Improvement: How much human feedback ratings improve during adaptation
- Desirable Improvement: Increase in desirable output generation
- Undesirable Reduction: Decrease in undesirable output generation
- Convergence Speed: How quickly the model adapts to preferences
- Consistency: Reliability across scenarios and seeds

Ranking uses Risk-Adjusted Score = Mean - Std to balance performance and reliability.

Usage:
    python scripts/benchmark/analyze_hil.py \
        --input-dir outputs/benchmark/hil_inference/
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Execution order for consistent display
EXECUTION_ORDER = [
    "baseline",
    "baseline_cl",
    "baseline_per",
    "baseline_per_cl",
    "baseline_noisy",
    "baseline_noisy_cl",
    "baseline_per_noisy",
    "baseline_per_noisy_cl",
    "dueling",
    "dueling_cl",
    "dueling_per",
    "dueling_per_cl",
    "dueling_noisy",
    "dueling_noisy_cl",
    "dueling_per_noisy",
    "dueling_per_noisy_cl",
    "rainbow",
    "rainbow_cl",
]


def load_combined_summary(variant_dir: Path) -> Optional[dict]:
    """Load combined_summary.json from a variant directory."""
    summary_file = variant_dir / "combined_summary.json"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not load {summary_file}: {e}")
        return None


def load_detailed_results(variant_dir: Path) -> List[dict]:
    """Load all individual result.json files from a variant directory."""
    results = []
    results_dir = variant_dir / "results"

    if not results_dir.exists():
        return results

    # Iterate through scenario directories
    for scenario_dir in results_dir.iterdir():
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name

        # Iterate through seed directories
        for seed_dir in scenario_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            result_file = seed_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                        data["_scenario"] = scenario_name
                        data["_seed_dir"] = seed_dir.name
                        results.append(data)
                except (json.JSONDecodeError, IOError):
                    continue

    return results


def parse_variant_name(dir_name: str) -> Tuple[str, Optional[int]]:
    """Parse variant name and seed from directory name.

    Examples:
        'baseline_cl_s42' -> ('baseline_cl', 42)
        'rainbow_s42' -> ('rainbow', 42)
    """
    # Remove seed suffix if present
    if "_s" in dir_name:
        parts = dir_name.rsplit("_s", 1)
        variant = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            seed = None
    else:
        variant = dir_name
        seed = None

    return variant, seed


def compute_hil_statistics(input_dir: Path) -> Dict:
    """Compute comprehensive HIL statistics from all variants.

    Args:
        input_dir: Directory containing variant subdirectories

    Returns:
        Dictionary with computed statistics
    """
    stats = {
        "total_variants": 0,
        "total_scenarios": 0,
        "total_runs": 0,
        "variants": {},
        "scenarios": set(),
    }

    # Scan variant directories
    variant_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not variant_dirs:
        print(f"Warning: No variant directories found in {input_dir}")
        return stats

    print(f"Scanning {len(variant_dirs)} variant directories...")

    for variant_dir in variant_dirs:
        variant_name, seed = parse_variant_name(variant_dir.name)

        # Load combined summary
        summary = load_combined_summary(variant_dir)
        if not summary:
            print(f"  [SKIP] {variant_dir.name}: No combined_summary.json")
            continue

        # Load detailed results for additional metrics
        detailed_results = load_detailed_results(variant_dir)

        # Extract scenario-level metrics
        scenarios_data = summary.get("scenarios", {})
        if not scenarios_data:
            print(f"  [SKIP] {variant_dir.name}: No scenario data")
            continue

        # Collect metrics across all scenarios
        feedback_improvements = []
        desirable_improvements = []
        undesirable_reductions = []
        convergence_iterations = []

        for scenario_name, scenario_metrics in scenarios_data.items():
            stats["scenarios"].add(scenario_name)

            # Get mean improvements (already aggregated across seeds in summary)
            fb_imp = scenario_metrics.get("feedback_improvement_mean")
            des_imp = scenario_metrics.get("desirable_improvement_mean")
            und_red = scenario_metrics.get("undesirable_reduction_mean")

            if fb_imp is not None:
                feedback_improvements.append(fb_imp)
            if des_imp is not None:
                desirable_improvements.append(des_imp)
            if und_red is not None:
                undesirable_reductions.append(und_red)

        # Get convergence iterations from detailed results
        for result in detailed_results:
            conv_iter = result.get("convergence_iteration")
            if conv_iter is not None:
                convergence_iterations.append(conv_iter)

        if not feedback_improvements:
            print(f"  [SKIP] {variant_dir.name}: No feedback improvement data")
            continue

        # Compute aggregate statistics
        mean_feedback = statistics.mean(feedback_improvements)
        std_feedback = statistics.stdev(feedback_improvements) if len(feedback_improvements) > 1 else 0.0
        risk_adjusted_feedback = mean_feedback - std_feedback

        mean_desirable = statistics.mean(desirable_improvements) if desirable_improvements else 0.0
        std_desirable = statistics.stdev(desirable_improvements) if len(desirable_improvements) > 1 else 0.0

        mean_undesirable = statistics.mean(undesirable_reductions) if undesirable_reductions else 0.0

        mean_convergence = statistics.mean(convergence_iterations) if convergence_iterations else 1000
        std_convergence = statistics.stdev(convergence_iterations) if len(convergence_iterations) > 1 else 0.0

        variant_stat = {
            "variant": variant_name,
            "dir_name": variant_dir.name,
            "n_scenarios": len(feedback_improvements),
            "n_runs": len(detailed_results),
            # Feedback metrics (primary)
            "mean_feedback_improvement": mean_feedback,
            "std_feedback_improvement": std_feedback,
            "risk_adjusted_feedback": risk_adjusted_feedback,
            # Desirable/Undesirable metrics
            "mean_desirable_improvement": mean_desirable,
            "std_desirable_improvement": std_desirable,
            "mean_undesirable_reduction": mean_undesirable,
            # Convergence metrics
            "mean_convergence_iteration": mean_convergence,
            "std_convergence_iteration": std_convergence,
            # Per-scenario breakdown
            "scenario_feedback": dict(zip(scenarios_data.keys(), feedback_improvements)),
            "raw_feedback_improvements": feedback_improvements,
        }

        stats["variants"][variant_name] = variant_stat
        stats["total_variants"] += 1
        stats["total_runs"] += len(detailed_results)

        print(f"  [OK] {variant_dir.name}: feedback_imp={mean_feedback:.4f}±{std_feedback:.4f}, "
              f"risk_adj={risk_adjusted_feedback:.4f}")

    stats["total_scenarios"] = len(stats["scenarios"])
    stats["scenarios"] = sorted(stats["scenarios"])

    # Compute rankings by risk-adjusted feedback score
    variant_list = [(v, s["risk_adjusted_feedback"]) for v, s in stats["variants"].items()]
    variant_list.sort(key=lambda x: x[1], reverse=True)

    for rank, (variant, _) in enumerate(variant_list, 1):
        stats["variants"][variant]["rank"] = rank

    return stats


def generate_hil_report(stats: Dict, output_dir: Path, input_dir: Path) -> str:
    """Generate markdown report for HIL analysis.

    Args:
        stats: Computed statistics
        output_dir: Output directory
        input_dir: Input directory (for reference)

    Returns:
        Markdown report content
    """
    report = f"""# HIL (Human-in-the-Loop) Inference Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Input Directory:** `{input_dir}`
**Analysis Type:** HIL Preference Adaptation Performance

## Summary

| Metric | Value |
|--------|-------|
| Variants Analyzed | {stats['total_variants']} |
| Scenarios | {stats['total_scenarios']} |
| Total Runs | {stats['total_runs']} |

### Scenarios Evaluated

{', '.join(f'`{s}`' for s in stats['scenarios'])}

## HIL Adaptation Performance

*Ranked by Risk-Adjusted Feedback Score = Mean Feedback Improvement - Std (higher is better). This balances adaptation quality with consistency across scenarios.*

"""

    if stats["variants"]:
        report += "| Method | Scenarios | Mean FB Imp | Std | Risk-Adj | Conv. Iter | Rank |\n"
        report += "|--------|-----------|-------------|-----|----------|------------|------|\n"

        # Sort by rank
        sorted_variants = sorted(
            stats["variants"].items(),
            key=lambda x: x[1].get("rank", 999)
        )

        for variant, v_stats in sorted_variants:
            rank = v_stats.get("rank", "N/A")
            rank_marker = "**1st**" if rank == 1 else f"{rank}th"
            bold = "**" if rank == 1 else ""

            # Build descriptive method name
            if "rainbow" in variant:
                method_name = "Rainbow DRQN"
            elif "dueling" in variant:
                method_name = "Dueling DRQN"
            else:
                method_name = "Baseline DRQN"

            components = []
            if "_per" in variant:
                components.append("PER")
            if "_noisy" in variant:
                components.append("Noisy")
            if components:
                method_name += " + " + " + ".join(components)

            cl_status = " (CL)" if "_cl" in variant else ""
            method_name += cl_status

            report += (
                f"| {bold}{method_name}{bold} | {v_stats['n_scenarios']} | "
                f"{bold}{v_stats['mean_feedback_improvement']:.4f}{bold} | "
                f"{v_stats['std_feedback_improvement']:.4f} | "
                f"{bold}{v_stats['risk_adjusted_feedback']:.4f}{bold} | "
                f"{v_stats['mean_convergence_iteration']:.0f} | {rank_marker} |\n"
            )

    report += "\n## Key Findings\n\n"

    # Winner summary
    if stats["variants"]:
        best = min(stats["variants"].items(), key=lambda x: x[1].get("rank", 999))
        report += f"**Best HIL Adapter:** {best[0]} "
        report += f"(Risk-Adj: {best[1]['risk_adjusted_feedback']:.4f} = "
        report += f"{best[1]['mean_feedback_improvement']:.4f} - {best[1]['std_feedback_improvement']:.4f})\n\n"

        # Fastest convergence
        fastest = min(stats["variants"].items(), key=lambda x: x[1].get("mean_convergence_iteration", 9999))
        report += f"**Fastest Convergence:** {fastest[0]} "
        report += f"(Mean: {fastest[1]['mean_convergence_iteration']:.0f} iterations)\n\n"

    # Detailed breakdown by metric
    report += "### Desirable Output Improvement\n\n"
    report += "*How much each method increases generation of desirable outputs.*\n\n"
    report += "| Method | Mean Improvement | Std |\n"
    report += "|--------|------------------|-----|\n"

    sorted_by_desirable = sorted(
        stats["variants"].items(),
        key=lambda x: x[1].get("mean_desirable_improvement", 0),
        reverse=True
    )

    for variant, v_stats in sorted_by_desirable[:5]:  # Top 5
        report += f"| {variant} | {v_stats['mean_desirable_improvement']:.4f} | {v_stats['std_desirable_improvement']:.4f} |\n"

    report += "\n### Convergence Speed\n\n"
    report += "*How quickly each method adapts to preferences (lower is better).*\n\n"
    report += "| Method | Mean Iterations | Std |\n"
    report += "|--------|-----------------|-----|\n"

    sorted_by_convergence = sorted(
        stats["variants"].items(),
        key=lambda x: x[1].get("mean_convergence_iteration", 9999)
    )

    for variant, v_stats in sorted_by_convergence[:5]:  # Top 5
        report += f"| {variant} | {v_stats['mean_convergence_iteration']:.0f} | {v_stats['std_convergence_iteration']:.0f} |\n"

    # Per-scenario breakdown
    report += "\n## Per-Scenario Performance\n\n"
    report += "*Feedback improvement by scenario for each method.*\n\n"

    # Create scenario comparison table
    scenarios = stats["scenarios"]
    if scenarios and stats["variants"]:
        header = "| Method |" + " | ".join(scenarios) + " |\n"
        separator = "|--------|" + " | ".join(["-----"] * len(scenarios)) + " |\n"
        report += header + separator

        for variant, v_stats in sorted_variants[:5]:  # Top 5 methods
            row = f"| {variant} |"
            scenario_fb = v_stats.get("scenario_feedback", {})
            for scenario in scenarios:
                val = scenario_fb.get(scenario, 0)
                row += f" {val:.3f} |"
            report += row + "\n"

    return report


def analyze_hil_results(
    input_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Main HIL analysis function.

    Args:
        input_dir: Directory containing HIL inference results
        output_dir: Output directory for analysis

    Returns:
        Dictionary with analysis results
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    print(f"\n{'='*60}")
    print("HIL INFERENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")

    # Compute statistics
    stats = compute_hil_statistics(input_dir)

    if not stats["variants"]:
        print("\nNo results found to analyze")
        return {}

    # Set output directory
    if output_dir is None:
        output_dir = input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {output_dir}")

    # Generate report
    report_content = generate_hil_report(stats, output_dir, input_dir)

    # Save outputs
    report_file = output_dir / "HIL_ANALYSIS_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"\n[OK] Report: {report_file}")

    # Save statistics JSON
    stats_file = output_dir / "hil_analysis_results.json"
    with open(stats_file, "w") as f:
        # Convert to JSON-serializable format
        json_stats = {
            "total_variants": stats["total_variants"],
            "total_scenarios": stats["total_scenarios"],
            "total_runs": stats["total_runs"],
            "scenarios": stats["scenarios"],
            "variants": {
                k: {kk: vv for kk, vv in v.items() if kk not in ["raw_feedback_improvements", "scenario_feedback"]}
                for k, v in stats["variants"].items()
            }
        }
        json.dump(json_stats, f, indent=2)
    print(f"[OK] Statistics: {stats_file}")

    # Save CSV
    csv_file = output_dir / "hil_method_comparison.csv"
    with open(csv_file, "w") as f:
        f.write("variant,base_method,per,noisy,cl,n_scenarios,mean_feedback_imp,std_feedback_imp,risk_adjusted,mean_convergence,rank\n")
        for variant, v_stats in stats["variants"].items():
            if "rainbow" in variant:
                base = "rainbow"
            elif "dueling" in variant:
                base = "dueling"
            else:
                base = "baseline"
            per = "yes" if "_per" in variant else "no"
            noisy = "yes" if "_noisy" in variant else "no"
            cl = "yes" if "_cl" in variant else "no"
            f.write(
                f"{variant},{base},{per},{noisy},{cl},"
                f"{v_stats['n_scenarios']},{v_stats['mean_feedback_improvement']:.4f},"
                f"{v_stats['std_feedback_improvement']:.4f},{v_stats['risk_adjusted_feedback']:.4f},"
                f"{v_stats['mean_convergence_iteration']:.0f},{v_stats.get('rank', 'N/A')}\n"
            )
    print(f"[OK] CSV: {csv_file}")

    return {
        "stats": stats,
        "report_file": str(report_file),
        "stats_file": str(stats_file),
        "csv_file": str(csv_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HIL inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes Human-in-the-Loop (HIL) preference adaptation results
to understand how models perform in deployment, beyond training benchmarks.

Key metrics:
- Feedback Improvement: How much human feedback ratings improve
- Convergence Speed: How quickly the model adapts
- Consistency: Reliability across scenarios

Examples:
    python scripts/nips_benchmark/analyze_hil_results.py \\
        --input-dir outputs/benchmark_reports/nips_benchmark/20251217_092325/hil_inference/
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing HIL inference results",
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
        result = analyze_hil_results(input_dir, output_dir)

        print(f"\n{'='*60}")
        print("HIL ANALYSIS COMPLETE")
        print(f"{'='*60}")

        if result:
            stats = result["stats"]
            print(f"Variants: {stats['total_variants']}")
            print(f"Scenarios: {stats['total_scenarios']}")
            print(f"Total Runs: {stats['total_runs']}")

            if stats["variants"]:
                best = min(stats["variants"].items(), key=lambda x: x[1].get("rank", 999))
                print(f"\nBest HIL Adapter: {best[0]}")
                print(f"  Risk-Adjusted Score: {best[1]['risk_adjusted_feedback']:.4f}")
                print(f"  Mean Feedback Improvement: {best[1]['mean_feedback_improvement']:.4f}")

            print(f"\nFull report: {result['report_file']}")

        print(f"{'='*60}")

    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
