#!/usr/bin/env python3
"""
Generate All Tables for NeurIPS Paper

This script generates all LaTeX tables needed for the NeurIPS paper
from training run data. It creates publication-ready tables that can
be directly copied into the paper.

Usage:
    python generate_paper_tables.py --input-dir artifacts/training --output-dir outputs/paper_tables
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from latex_tables import LatexTableGenerator, AblationAnalyzer
from analyze_training_runs import (
    load_run_data,
    compute_statistics,
    AnalysisConfig,
    RunMetrics,
)


def load_all_runs(input_dir: Path, analysis_config: AnalysisConfig) -> List[RunMetrics]:
    """Load all training runs from directory."""
    run_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    print(f"Found {len(run_dirs)} training runs")

    runs = []
    for run_dir in run_dirs:
        run_data = load_run_data(run_dir)
        if run_data:
            run_data = compute_statistics(run_data, analysis_config)
            runs.append(run_data)
            print(f"  Loaded: {run_data.name}")

    print(f"\nSuccessfully loaded {len(runs)} runs")
    return runs


def filter_runs_by_pattern(runs: List[RunMetrics], patterns: List[str]) -> List[RunMetrics]:
    """Filter runs by name patterns."""
    filtered = []
    for run in runs:
        if any(pattern in run.name.lower() for pattern in patterns):
            filtered.append(run)
    return filtered


def generate_all_tables(
    runs: List[RunMetrics],
    output_dir: Path,
    generator: LatexTableGenerator,
) -> Dict[str, str]:
    """Generate all tables for the paper."""

    tables = {}

    # =========================================================================
    # TABLE 1: Training Performance Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 1: Training Performance Comparison")
    print("="*70)

    # Filter for baseline comparison runs
    baseline_patterns = [
        'end-to-end', 'single-agent', 'joint-training',
        'vanilla-drqn', 'rainbow-drqn', 'no-curriculum'
    ]
    baseline_runs = filter_runs_by_pattern(runs, baseline_patterns)

    if baseline_runs:
        tables['tab1_training_comparison'] = generator.generate_training_comparison_table(
            runs=baseline_runs,
            baseline_name='end-to-end',
            caption="Training performance comparison. Mean $\\pm$ std over 5 seeds.",
            label="tab:training_results",
        )
        print(f"  Generated with {len(baseline_runs)} runs")
    else:
        print("  WARNING: No baseline runs found")

    # =========================================================================
    # TABLE 2: Coordination Mechanism Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 2: Coordination Mechanism Ablation")
    print("="*70)

    coord_patterns = [
        'prototype', 'centroid', 'random-action', 'one-hot',
        'no-topology', 'online-perceiving'
    ]
    coord_runs = filter_runs_by_pattern(runs, coord_patterns)

    if coord_runs:
        tables['tab2_coordination_ablation'] = generator.generate_ablation_table(
            runs=coord_runs,
            full_method_name='prototype',
            components=['centroid', 'random-action', 'one-hot', 'topology', 'online'],
            caption="Ablation: coordination mechanisms. Relative to full ARIA with prototype embeddings.",
            label="tab:coordination_ablation",
        )
        print(f"  Generated with {len(coord_runs)} runs")
    else:
        print("  WARNING: No coordination ablation runs found")

    # =========================================================================
    # TABLE 3: Curriculum Learning Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 3: Curriculum Learning Ablation")
    print("="*70)

    curriculum_patterns = ['no-curriculum', '2-stage', '3-stage', 'manual-curriculum']
    curriculum_runs = filter_runs_by_pattern(runs, curriculum_patterns)

    if curriculum_runs:
        tables['tab3_curriculum_ablation'] = generator.generate_ablation_table(
            runs=curriculum_runs,
            full_method_name='3-stage',
            components=['curriculum'],
            caption="Ablation: curriculum learning via GHSOM hierarchy.",
            label="tab:curriculum_ablation",
        )
        print(f"  Generated with {len(curriculum_runs)} runs")
    else:
        print("  WARNING: No curriculum ablation runs found")

    # =========================================================================
    # TABLE 4: Rainbow Component Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 4: Rainbow Component Ablation")
    print("="*70)

    rainbow_patterns = [
        'full-rainbow', 'no-c51', 'no-prioritized', 'no-burnin',
        'no-dueling', 'no-double', 'no-nstep', 'no-noisy'
    ]
    rainbow_runs = filter_runs_by_pattern(runs, rainbow_patterns)

    if rainbow_runs:
        tables['tab4_rainbow_ablation'] = generator.generate_ablation_table(
            runs=rainbow_runs,
            full_method_name='full-rainbow',
            components=['c51', 'prioritized', 'burnin', 'dueling', 'double', 'nstep', 'noisy'],
            caption="Ablation: Rainbow DRQN components.",
            label="tab:rainbow_ablation",
        )
        print(f"  Generated with {len(rainbow_runs)} runs")
    else:
        print("  WARNING: No Rainbow ablation runs found")

    # =========================================================================
    # TABLE 5: Feature Extraction Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 5: Feature Extraction Ablation")
    print("="*70)

    feature_patterns = [
        'full-features', 'no-prettymidi', 'no-muspy',
        'no-custom', 'random-features'
    ]
    feature_runs = filter_runs_by_pattern(runs, feature_patterns)

    if feature_runs:
        tables['tab5_feature_ablation'] = generator.generate_ablation_table(
            runs=feature_runs,
            full_method_name='full-features',
            components=['prettymidi', 'muspy', 'custom'],
            caption="Ablation: feature extraction sources. Impact on final policy reward.",
            label="tab:feature_ablation",
        )
        print(f"  Generated with {len(feature_runs)} runs")
    else:
        print("  WARNING: No feature ablation runs found")

    # =========================================================================
    # TABLE 6: Reward Component Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 6: Reward Component Ablation")
    print("="*70)

    reward_patterns = [
        'full-reward', 'no-structure', 'no-transition',
        'no-diversity', 'terminal-only', 'incremental'
    ]
    reward_runs = filter_runs_by_pattern(runs, reward_patterns)

    if reward_runs:
        tables['tab6_reward_ablation'] = generator.generate_ablation_table(
            runs=reward_runs,
            full_method_name='full-reward',
            components=['structure', 'transition', 'diversity'],
            caption="Ablation: reward function components and credit assignment.",
            label="tab:reward_ablation",
        )
        print(f"  Generated with {len(reward_runs)} runs")
    else:
        print("  WARNING: No reward ablation runs found")

    # =========================================================================
    # TABLE 7: Learning Rate Schedule Ablation
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 7: Learning Rate Schedule Ablation")
    print("="*70)

    lr_patterns = ['constant', 'exponential', 'linear', 'lr']
    lr_runs = filter_runs_by_pattern(runs, lr_patterns)

    if lr_runs:
        tables['tab7_lr_ablation'] = generator.generate_lr_schedule_table(
            runs=lr_runs,
            caption="Ablation: learning rate schedule.",
            label="tab:lr_ablation",
        )
        print(f"  Generated with {len(lr_runs)} runs")
    else:
        print("  WARNING: No LR schedule runs found")

    # =========================================================================
    # TABLE 8: Diversity Range Grid Search
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Table 8: Diversity Range Grid Search")
    print("="*70)

    diversity_patterns = ['diversity']
    diversity_runs = filter_runs_by_pattern(runs, diversity_patterns)

    if diversity_runs:
        # Custom diversity table (simplified)
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Grid search: diversity reward range optimization.}")
        lines.append("\\label{tab:diversity_grid}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Diversity Range} & \\textbf{Unique Clusters} & \\textbf{Reward} \\\\")
        lines.append("\\midrule")

        # Sort by performance
        sorted_divs = sorted(diversity_runs, key=lambda r: r.final_mean_reward, reverse=True)

        for i, run in enumerate(sorted_divs):
            # Extract range from metadata or config
            div_range = run.metadata.get('diversity_range', '[TBD]')
            clusters = run.metadata.get('unique_clusters', '[TBD]')
            reward = run.final_mean_reward

            if isinstance(div_range, list) and len(div_range) == 2:
                range_str = f"$[{div_range[0]:.2f}, {div_range[1]:.2f}]$"
            else:
                range_str = str(div_range)

            if i == 0:  # Best result
                range_str = f"$\\mathbf{{{range_str[1:-1]}}}$"
                clusters = f"\\textbf{{{clusters}}}"
                reward_str = f"$\\mathbf{{{reward:.2f}}}$"
            else:
                reward_str = f"${reward:.2f}$"

            lines.append(f"{range_str} & {clusters} & {reward_str} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        tables['tab8_diversity_grid'] = "\n".join(lines)
        print(f"  Generated with {len(diversity_runs)} runs")
    else:
        print("  WARNING: No diversity grid search runs found")

    return tables


def save_tables(tables: Dict[str, str], output_dir: Path):
    """Save all tables to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Saving Tables")
    print("="*70)

    for name, latex_code in tables.items():
        output_file = output_dir / f"{name}.tex"
        output_file.write_text(latex_code)
        print(f"  Saved: {output_file}")

    # Also save a combined file
    combined_file = output_dir / "all_tables.tex"
    with open(combined_file, 'w') as f:
        f.write("% All Tables for NeurIPS Paper\n")
        f.write("% Generated by generate_paper_tables.py\n\n")

        for name, latex_code in tables.items():
            f.write(f"% {name}\n")
            f.write(latex_code)
            f.write("\n\n")

    print(f"  Saved combined: {combined_file}")

    # Save metadata
    metadata = {
        'num_tables': len(tables),
        'table_names': list(tables.keys()),
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_file}")


def print_summary(tables: Dict[str, str]):
    """Print summary of generated tables."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nGenerated {len(tables)} tables:")

    for i, name in enumerate(tables.keys(), 1):
        print(f"  {i}. {name}")

    print("\nTo use in your paper:")
    print("  1. Copy the .tex files to your paper directory")
    print("  2. Ensure \\usepackage{booktabs} is in your preamble")
    print("  3. Use \\input{<table_file>} to include tables")
    print("  4. Or copy-paste the LaTeX code directly")

    print("\nNext steps:")
    print("  - Review tables for correctness")
    print("  - Replace [TBD] placeholders with actual values")
    print("  - Adjust captions and labels as needed")
    print("  - Verify table formatting in compiled PDF")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate all LaTeX tables for NeurIPS paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('artifacts/training'),
        help='Directory containing training runs',
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/paper_tables'),
        help='Output directory for LaTeX tables',
    )

    parser.add_argument(
        '--decimals',
        type=int,
        default=2,
        help='Number of decimal places for formatting',
    )

    parser.add_argument(
        '--no-bold',
        action='store_true',
        help='Disable automatic bolding of best values',
    )

    args = parser.parse_args()

    # Check input directory
    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        print("Please provide a valid directory with training runs.")
        return 1

    # Load runs
    analysis_config = AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=Path('outputs'),
    )

    runs = load_all_runs(args.input_dir, analysis_config)

    if not runs:
        print("\nERROR: No valid training runs found!")
        print("Please ensure the input directory contains valid run directories.")
        return 1

    # Initialize generator
    generator = LatexTableGenerator(
        decimals=args.decimals,
        use_booktabs=True,
        bold_best=not args.no_bold,
        show_delta=True,
    )

    # Generate tables
    tables = generate_all_tables(runs, args.output_dir, generator)

    if not tables:
        print("\nWARNING: No tables were generated!")
        print("This might be because run names don't match expected patterns.")
        return 1

    # Save tables
    save_tables(tables, args.output_dir)

    # Print summary
    print_summary(tables)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
