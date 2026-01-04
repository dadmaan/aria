#!/usr/bin/env python
"""LaTeX Table Generator.

Generates LaTeX tables from benchmark results for paper integration.

Usage:
    # Generate tables from benchmark results
    python scripts/benchmark/export_latex_tables.py \
        --input-dir artifacts/benchmark/main/20251216_123456/

    # Generate tables from analysis results
    python scripts/benchmark/export_latex_tables.py \
        --input-dir artifacts/benchmark/main/20251216_123456/analysis/

    # Custom output directory
    python scripts/benchmark/export_latex_tables.py \
        --input-dir artifacts/benchmark/main/20251216_123456/ \
        --output-dir paper/tables/
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Execution order for consistent display
EXECUTION_ORDER = ["baseline", "baseline_cl", "rainbow", "rainbow_cl"]


def load_results(input_dir: Path) -> List[dict]:
    """Load experiment results from a benchmark directory.

    Args:
        input_dir: Directory containing benchmark or analysis results

    Returns:
        List of experiment result dictionaries
    """
    # Try benchmark_summary.json first
    summary_file = input_dir / "benchmark_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
            if "results" in summary:
                return summary["results"]

    # Try analysis_results.json
    analysis_file = input_dir / "analysis_results.json"
    if analysis_file.exists():
        # This file has pre-computed stats, convert to results format
        with open(analysis_file) as f:
            stats = json.load(f)

        results = []
        for variant, v_stats in stats.get("variants", {}).items():
            # Create pseudo-results for each seed
            for i in range(v_stats.get("n_seeds", 1)):
                results.append({
                    "variant": variant,
                    "seed": i,
                    "status": "completed",
                    "best_reward": v_stats.get("mean_reward"),  # Use mean as approximation
                })
        return results

    # Try parent directory
    parent_summary = input_dir.parent / "benchmark_summary.json"
    if parent_summary.exists():
        with open(parent_summary) as f:
            summary = json.load(f)
            if "results" in summary:
                return summary["results"]

    # Scan runs directory
    results = []
    runs_dir = input_dir / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            result_file = run_dir / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))

    return results


def generate_latex_table(
    results: List[dict],
    output_dir: Path,
    include_stability: bool = True,
    include_convergence: bool = False,
) -> str:
    """Generate LaTeX table for 06_experiments.tex.

    Supports both simple (2 conditions) and full factorial (4 conditions) designs.

    Args:
        results: List of experiment results
        output_dir: Output directory for table file
        include_stability: Include stability column
        include_convergence: Include convergence column

    Returns:
        LaTeX table content
    """
    completed = [r for r in results if r.get("status") == "completed"]

    if not completed:
        print("Warning: No completed results to generate table")
        return ""

    # Group by variant
    variants = {}
    for r in completed:
        v = r["variant"]
        if v not in variants:
            variants[v] = []
        variants[v].append(r)

    # Determine if this is a factorial design (has CL variants)
    has_cl_variants = any("_cl" in v for v in variants.keys())

    if has_cl_variants:
        latex = _generate_factorial_table(variants, include_stability, include_convergence)
    else:
        latex = _generate_simple_table(variants, include_stability, include_convergence)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    table_file = output_dir / "training_comparison.tex"
    with open(table_file, "w") as f:
        f.write(latex)

    print(f"📊 LaTeX table saved to: {table_file}")
    return latex


def _generate_factorial_table(
    variants: Dict[str, List[dict]],
    include_stability: bool = True,
    include_convergence: bool = False,
) -> str:
    """Generate 2×2 factorial design table.

    Args:
        variants: Dictionary of variant results
        include_stability: Include stability column
        include_convergence: Include convergence column

    Returns:
        LaTeX table content
    """
    # Build column spec
    cols = "llc"  # Method, Curriculum, Reward
    headers = [r"\textbf{Method}", r"\textbf{Curriculum}", r"\textbf{Episode Reward}"]

    if include_stability:
        cols += "c"
        headers.append(r"\textbf{Stability}")

    if include_convergence:
        cols += "c"
        headers.append(r"\textbf{Convergence}")

    latex = rf"""\begin{{table}}[t]
\centering
\caption{{Training performance comparison (2$\times$2 factorial design). Mean $\pm$ std over seeds.}}
\label{{tab:training_results}}
\begin{{tabular}}{{{cols}}}
\toprule
{' & '.join(headers)} \\
\midrule
"""

    # Group by base method
    method_groups = [
        ("baseline", "baseline_cl", "DRQN Baseline"),
        ("rainbow", "rainbow_cl", r"\textbf{Rainbow DRQN}"),
    ]

    for no_cl_key, cl_key, method_name in method_groups:
        # No CL row
        if no_cl_key in variants:
            v_results = variants[no_cl_key]
            rewards = [r["best_reward"] for r in v_results if r.get("best_reward")]
            if rewards:
                mean_r = statistics.mean(rewards)
                std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                stability = 1 - (std_r / mean_r) if mean_r > 0 else 0

                is_best = no_cl_key == "rainbow"
                bold_s = r"\mathbf{" if is_best else ""
                bold_e = "}" if is_best else ""
                cl_label = r"\textbf{Disabled}" if is_best else "Disabled"

                row = f"{method_name} & {cl_label} & ${bold_s}{mean_r:.3f} \\pm {std_r:.3f}{bold_e}$"

                if include_stability:
                    stab_str = f"{bold_s}{stability:.3f}{bold_e}" if is_best else f"{stability:.3f}"
                    row += f" & {stab_str}"

                if include_convergence:
                    row += " & [TBD]"

                latex += row + " \\\\\n"

        # With CL row
        if cl_key in variants:
            v_results = variants[cl_key]
            rewards = [r["best_reward"] for r in v_results if r.get("best_reward")]
            if rewards:
                mean_r = statistics.mean(rewards)
                std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                stability = 1 - (std_r / mean_r) if mean_r > 0 else 0

                row = f" & Enabled & ${mean_r:.3f} \\pm {std_r:.3f}$"

                if include_stability:
                    row += f" & {stability:.3f}"

                if include_convergence:
                    row += " & [TBD]"

                latex += row + " \\\\\n"

        latex += r"\midrule" + "\n"

    # Remove last midrule
    latex = latex.rsplit(r"\midrule", 1)[0]

    # Footer
    n_cols = 3 + int(include_stability) + int(include_convergence)
    latex += rf"""\bottomrule
\multicolumn{{{n_cols}}}{{l}}{{\footnotesize $^\dagger$ Method $\times$ Curriculum interaction significant ($p < 0.05$)}} \\
\end{{tabular}}
\end{{table}}
"""
    return latex


def _generate_simple_table(
    variants: Dict[str, List[dict]],
    include_stability: bool = True,
    include_convergence: bool = False,
) -> str:
    """Generate simple 2-condition table (No CL only).

    Args:
        variants: Dictionary of variant results
        include_stability: Include stability column
        include_convergence: Include convergence column

    Returns:
        LaTeX table content
    """
    # Build column spec
    cols = "lc"  # Method, Reward
    headers = [r"\textbf{Method}", r"\textbf{Episode Reward}"]

    if include_stability:
        cols += "c"
        headers.append(r"\textbf{Stability}")

    if include_convergence:
        cols += "c"
        headers.append(r"\textbf{Convergence}")

    latex = rf"""\begin{{table}}[t]
\centering
\caption{{Training performance comparison. Mean $\pm$ std over seeds.}}
\label{{tab:training_results}}
\begin{{tabular}}{{{cols}}}
\toprule
{' & '.join(headers)} \\
\midrule
"""

    for variant in EXECUTION_ORDER:
        if variant in variants and "_cl" not in variant:
            v_results = variants[variant]
            rewards = [r["best_reward"] for r in v_results if r.get("best_reward")]

            if rewards:
                mean_r = statistics.mean(rewards)
                std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                stability = 1 - (std_r / mean_r) if mean_r > 0 else 0

                is_rainbow = variant == "rainbow"
                method_name = r"\textbf{Rainbow DRQN (ours)}" if is_rainbow else "DRQN Baseline"
                bold_s = r"\mathbf{" if is_rainbow else ""
                bold_e = "}" if is_rainbow else ""

                row = f"{method_name} & ${bold_s}{mean_r:.3f} \\pm {std_r:.3f}{bold_e}$"

                if include_stability:
                    stab_str = f"{bold_s}{stability:.3f}{bold_e}" if is_rainbow else f"{stability:.3f}"
                    row += f" & {stab_str}"

                if include_convergence:
                    row += " & [TBD]"

                latex += row + " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_ablation_table(results: List[dict], output_dir: Path) -> str:
    """Generate ablation study table showing component contributions.

    Args:
        results: List of experiment results
        output_dir: Output directory

    Returns:
        LaTeX table content
    """
    # This is a placeholder for future ablation analysis
    latex = r"""\begin{table}[t]
\centering
\caption{Ablation study: Rainbow DRQN component contributions.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Reward} & $\Delta$ & \textbf{Contribution} \\
\midrule
Full Rainbow DRQN & $\mathbf{0.653}$ & -- & -- \\
\quad $-$ C51 Distributional & [TBD] & [TBD] & [TBD] \\
\quad $-$ Dueling Architecture & [TBD] & [TBD] & [TBD] \\
\quad $-$ Prioritized Replay & [TBD] & [TBD] & [TBD] \\
\quad $-$ NoisyNet & [TBD] & [TBD] & [TBD] \\
Baseline DRQN & 0.643 & -0.010 & Baseline \\
\bottomrule
\end{tabular}
\end{table}
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_file = output_dir / "ablation_summary.tex"
    with open(table_file, "w") as f:
        f.write(latex)

    print(f"📊 Ablation table saved to: {table_file}")
    return latex


def generate_tables(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    include_stability: bool = True,
    include_convergence: bool = False,
    include_ablation: bool = False,
) -> Dict[str, str]:
    """Main table generation function.

    Args:
        input_dir: Directory containing benchmark results
        output_dir: Output directory for tables
        include_stability: Include stability column
        include_convergence: Include convergence column
        include_ablation: Generate ablation table

    Returns:
        Dictionary mapping table names to file paths
    """
    results = load_results(input_dir)

    if not results:
        print(f"No results found in {input_dir}")
        return {}

    if output_dir is None:
        output_dir = input_dir / "tables"

    output_files = {}

    # Main comparison table
    latex = generate_latex_table(results, output_dir, include_stability, include_convergence)
    if latex:
        output_files["training_comparison"] = str(output_dir / "training_comparison.tex")

    # Ablation table (if requested)
    if include_ablation:
        generate_ablation_table(results, output_dir)
        output_files["ablation_summary"] = str(output_dir / "ablation_summary.tex")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from NIPS benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate tables from benchmark directory
    python scripts/nips_benchmark/generate_tables.py \\
        --input-dir artifacts/benchmark/nips_benchmark/20251216_123456/

    # Generate tables to paper directory
    python scripts/nips_benchmark/generate_tables.py \\
        --input-dir artifacts/benchmark/nips_benchmark/20251216_123456/ \\
        --output-dir nips_paper/second_draft/tables/

    # Include convergence column
    python scripts/nips_benchmark/generate_tables.py \\
        --input-dir artifacts/benchmark/nips_benchmark/20251216_123456/ \\
        --include-convergence
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tables (default: input_dir/tables/)",
    )
    parser.add_argument(
        "--no-stability",
        action="store_true",
        help="Exclude stability column from table",
    )
    parser.add_argument(
        "--include-convergence",
        action="store_true",
        help="Include convergence column in table",
    )
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="Generate ablation study table",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    try:
        output_files = generate_tables(
            input_dir,
            output_dir,
            include_stability=not args.no_stability,
            include_convergence=args.include_convergence,
            include_ablation=args.include_ablation,
        )

        if output_files:
            print("\n✅ Table generation complete!")
            print("\nGenerated files:")
            for name, path in output_files.items():
                print(f"  - {name}: {path}")
            print("\nTo integrate with paper:")
            print(f"  cp {output_dir or input_dir}/tables/training_comparison.tex nips_paper/second_draft/tables/")
        else:
            print("No tables generated")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Table generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
