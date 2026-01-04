#!/usr/bin/env python3
"""
Example Usage of LaTeX Table Generation

Demonstrates how to use the LatexTableGenerator and AblationAnalyzer
to create publication-ready tables for the NeurIPS paper.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from latex_tables import (
    LatexTableGenerator,
    AblationAnalyzer,
    AblationGroup,
    generate_all_tables_from_runs,
)
from analyze_training_runs import RunMetrics, load_run_data, compute_statistics, AnalysisConfig

import numpy as np


def example_with_real_data():
    """Example using real training data if available."""
    input_dir = Path("../../artifacts/training")

    if not input_dir.exists():
        print(f"Training data not found at {input_dir}")
        print("Run the mock example instead.")
        return

    # Load runs
    run_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        print("No training runs found")
        return

    print(f"Found {len(run_dirs)} training runs")

    # Load data
    runs = []
    config = AnalysisConfig(input_dir=input_dir, output_dir=Path("outputs"))

    for run_dir in run_dirs[:5]:  # Load first 5 runs
        run_data = load_run_data(run_dir)
        if run_data:
            run_data = compute_statistics(run_data, config)
            runs.append(run_data)

    if not runs:
        print("No valid runs loaded")
        return

    print(f"Loaded {len(runs)} runs")

    # Generate tables
    generator = LatexTableGenerator(decimals=2, bold_best=True)

    # 1. Training Comparison Table
    print("\n" + "=" * 70)
    print("TABLE 1: TRAINING COMPARISON")
    print("=" * 70)
    table1 = generator.generate_training_comparison_table(
        runs=runs,
        baseline_name=runs[0].name,
        caption="Training performance comparison. Mean $\\pm$ std over 5 seeds.",
        label="tab:training_results",
    )
    print(table1)

    # 2. Ablation Table (if runs have different components)
    print("\n" + "=" * 70)
    print("TABLE 2: ABLATION STUDY")
    print("=" * 70)
    table2 = generator.generate_ablation_table(
        runs=runs,
        full_method_name=runs[0].name,
        components=["component_a", "component_b"],
        caption="Ablation study: component contributions.",
        label="tab:ablation",
    )
    print(table2)

    # 3. Learning Rate Schedule Table
    print("\n" + "=" * 70)
    print("TABLE 3: LEARNING RATE SCHEDULE")
    print("=" * 70)
    table3 = generator.generate_lr_schedule_table(
        runs=runs,
        caption="Ablation: learning rate schedule comparison.",
        label="tab:lr_ablation",
    )
    print(table3)

    # Use AblationAnalyzer
    print("\n" + "=" * 70)
    print("ABLATION ANALYSIS")
    print("=" * 70)

    analyzer = AblationAnalyzer(significance_threshold=0.05)

    # Group by learning rate schedule
    group = analyzer.group_by_dimension(
        runs=runs,
        dimension="training.learning_rate_scheduler.type",
    )

    print(f"Baseline: {group.baseline_run.name}")
    print(f"Variants: {[r.name for r in group.variant_runs]}")

    # Compute relative changes
    changes = analyzer.compute_relative_changes(group, metric="final_mean_reward")
    print("\nRelative changes:")
    for run_name, change in changes.items():
        print(f"  {run_name}: {change:+.2f}%")

    # Statistical significance
    if len(runs) >= 2:
        p_value, is_sig = analyzer.test_significance(runs[0], runs[1])
        print(f"\nStatistical test: p={p_value:.4f}, significant={is_sig}")

    # Save all tables
    output_dir = Path("outputs/latex_tables")
    generate_all_tables_from_runs(runs, output_dir)
    print(f"\n\nTables saved to: {output_dir}")


def example_with_mock_data():
    """Example with mock data for demonstration."""
    print("=" * 70)
    print("MOCK DATA EXAMPLE")
    print("=" * 70)

    # Create mock RunMetrics objects
    from dataclasses import dataclass, field
    from typing import Any, Dict

    @dataclass
    class MockRunMetrics:
        name: str
        final_mean_reward: float
        final_std_reward: float
        mean_reward: float = 0.0
        std_reward: float = 0.0
        min_reward: float = 0.0
        max_reward: float = 0.0
        trend_slope: float = 0.0
        stability_score: float = 0.8
        convergence_episode: int = None
        total_episodes: int = 1000
        total_steps: int = 50000
        episode_rewards: np.ndarray = field(default_factory=lambda: np.random.randn(1000))
        episode_lengths: np.ndarray = field(default_factory=lambda: np.ones(1000))
        config: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        anomalies: list = field(default_factory=list)
        tb_loss: list = field(default_factory=list)
        tb_lr: list = field(default_factory=list)
        tb_exploration_rate: list = field(default_factory=list)

    # Create sample runs for different ablation studies
    runs_training_comparison = [
        MockRunMetrics(
            name="run_end-to-end-dqn",
            final_mean_reward=0.45,
            final_std_reward=0.03,
            mean_reward=0.42,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
            metadata={'training_time_hours': 2.1},
        ),
        MockRunMetrics(
            name="run_single-agent-drqn",
            final_mean_reward=0.47,
            final_std_reward=0.04,
            mean_reward=0.45,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
            metadata={'training_time_hours': 2.0},
        ),
        MockRunMetrics(
            name="run_joint-training",
            final_mean_reward=0.49,
            final_std_reward=0.05,
            mean_reward=0.46,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
            metadata={'training_time_hours': 2.3},
        ),
        MockRunMetrics(
            name="run_vanilla-drqn",
            final_mean_reward=0.48,
            final_std_reward=0.03,
            mean_reward=0.46,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
            metadata={'training_time_hours': 1.9},
        ),
        MockRunMetrics(
            name="run_rainbow-drqn-ours",
            final_mean_reward=0.52,
            final_std_reward=0.02,
            mean_reward=0.50,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {
                        'enabled': True,
                        'type': 'exponential',
                        'initial_lr': 0.001,
                        'final_lr': 0.0001,
                    },
                }
            },
            metadata={'training_time_hours': 1.8},
        ),
    ]

    # Rainbow ablation
    runs_rainbow_ablation = [
        MockRunMetrics(
            name="run_full-rainbow-drqn",
            final_mean_reward=0.52,
            final_std_reward=0.02,
        ),
        MockRunMetrics(
            name="run_no-c51-distributional",
            final_mean_reward=0.48,
            final_std_reward=0.03,
        ),
        MockRunMetrics(
            name="run_no-prioritized-replay",
            final_mean_reward=0.49,
            final_std_reward=0.02,
        ),
        MockRunMetrics(
            name="run_no-8step-burnin",
            final_mean_reward=0.50,
            final_std_reward=0.03,
        ),
        MockRunMetrics(
            name="run_no-dueling-streams",
            final_mean_reward=0.49,
            final_std_reward=0.02,
        ),
        MockRunMetrics(
            name="run_no-double-dqn",
            final_mean_reward=0.48,
            final_std_reward=0.04,
        ),
        MockRunMetrics(
            name="run_no-nstep-returns",
            final_mean_reward=0.47,
            final_std_reward=0.03,
        ),
        MockRunMetrics(
            name="run_no-noisynet",
            final_mean_reward=0.50,
            final_std_reward=0.03,
        ),
    ]

    # LR schedule ablation
    runs_lr_schedule = [
        MockRunMetrics(
            name="run_constant-lr-1e-3",
            final_mean_reward=0.47,
            final_std_reward=0.03,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
        ),
        MockRunMetrics(
            name="run_constant-lr-1e-4",
            final_mean_reward=0.44,
            final_std_reward=0.02,
            config={
                'training': {
                    'learning_rate': 0.0001,
                    'learning_rate_scheduler': {'enabled': False},
                }
            },
        ),
        MockRunMetrics(
            name="run_exponential-decay",
            final_mean_reward=0.52,
            final_std_reward=0.02,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {
                        'enabled': True,
                        'type': 'exponential',
                        'initial_lr': 0.001,
                        'final_lr': 0.0001,
                    },
                }
            },
        ),
        MockRunMetrics(
            name="run_linear-decay",
            final_mean_reward=0.50,
            final_std_reward=0.03,
            config={
                'training': {
                    'learning_rate': 0.001,
                    'learning_rate_scheduler': {
                        'enabled': True,
                        'type': 'linear',
                        'initial_lr': 0.001,
                        'final_lr': 0.0001,
                    },
                }
            },
        ),
    ]

    # Diversity range grid search
    runs_diversity_grid = [
        MockRunMetrics(
            name="run_diversity-[0.50-0.62]",
            final_mean_reward=0.48,
            final_std_reward=0.03,
            metadata={'diversity_range': [0.50, 0.62], 'unique_clusters': '8-10'},
        ),
        MockRunMetrics(
            name="run_diversity-[0.56-0.69]",
            final_mean_reward=0.50,
            final_std_reward=0.02,
            metadata={'diversity_range': [0.56, 0.69], 'unique_clusters': '9-11'},
        ),
        MockRunMetrics(
            name="run_diversity-[0.62-0.75]",
            final_mean_reward=0.52,
            final_std_reward=0.02,
            metadata={'diversity_range': [0.62, 0.75], 'unique_clusters': '10-12'},
        ),
        MockRunMetrics(
            name="run_diversity-[0.69-0.81]",
            final_mean_reward=0.50,
            final_std_reward=0.03,
            metadata={'diversity_range': [0.69, 0.81], 'unique_clusters': '11-13'},
        ),
        MockRunMetrics(
            name="run_diversity-[0.75-0.88]",
            final_mean_reward=0.48,
            final_std_reward=0.04,
            metadata={'diversity_range': [0.75, 0.88], 'unique_clusters': '12-14'},
        ),
    ]

    generator = LatexTableGenerator(decimals=2, bold_best=True)

    # 1. Training Comparison
    print("\n" + "=" * 70)
    print("TABLE 1: TRAINING COMPARISON")
    print("=" * 70 + "\n")
    table1 = generator.generate_training_comparison_table(
        runs=runs_training_comparison,
        baseline_name="end-to-end-dqn",
        caption="Training performance comparison. Mean $\\pm$ std over 5 seeds.",
        label="tab:training_results",
    )
    print(table1)

    # 2. Rainbow Ablation
    print("\n" + "=" * 70)
    print("TABLE 2: RAINBOW COMPONENT ABLATION")
    print("=" * 70 + "\n")
    table2 = generator.generate_ablation_table(
        runs=runs_rainbow_ablation,
        full_method_name="full-rainbow-drqn",
        components=["c51", "prioritized-replay", "burnin", "dueling", "double-dqn", "nstep", "noisynet"],
        caption="Ablation: Rainbow DRQN components.",
        label="tab:rainbow_ablation",
    )
    print(table2)

    # 3. LR Schedule
    print("\n" + "=" * 70)
    print("TABLE 3: LEARNING RATE SCHEDULE ABLATION")
    print("=" * 70 + "\n")
    table3 = generator.generate_lr_schedule_table(
        runs=runs_lr_schedule,
        caption="Ablation: learning rate schedule.",
        label="tab:lr_ablation",
    )
    print(table3)

    # 4. Diversity Grid Search
    print("\n" + "=" * 70)
    print("TABLE 4: DIVERSITY RANGE GRID SEARCH")
    print("=" * 70 + "\n")

    # Custom table for diversity (simpler approach)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Grid search: diversity reward range optimization.}")
    lines.append("\\label{tab:diversity_grid}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Diversity Range} & \\textbf{Unique Clusters} & \\textbf{Reward} \\\\")
    lines.append("\\midrule")

    sorted_div = sorted(runs_diversity_grid, key=lambda r: r.final_mean_reward, reverse=True)
    for i, run in enumerate(sorted_div):
        div_range = run.metadata.get('diversity_range', [0, 0])
        clusters = run.metadata.get('unique_clusters', 'N/A')
        reward = run.final_mean_reward

        range_str = f"$[{div_range[0]:.2f}, {div_range[1]:.2f}]$"
        if i == 0:
            range_str = f"$\\mathbf{{[{div_range[0]:.2f}, {div_range[1]:.2f}]}}$"
            clusters = f"\\textbf{{{clusters}}}"
            reward_str = f"$\\mathbf{{{reward:.2f}}}$"
        else:
            reward_str = f"${reward:.2f}$"

        lines.append(f"{range_str} & {clusters} & {reward_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table4 = "\n".join(lines)
    print(table4)

    # Demonstrate AblationAnalyzer
    print("\n" + "=" * 70)
    print("ABLATION ANALYZER DEMO")
    print("=" * 70 + "\n")

    analyzer = AblationAnalyzer(significance_threshold=0.05)

    # Analyze Rainbow components
    group = analyzer.group_by_dimension(
        runs=runs_rainbow_ablation,
        dimension="name",
        baseline_value=None,  # Auto-detect best
    )

    print(f"Baseline: {group.baseline_run.name}")
    print(f"  Reward: {group.baseline_run.final_mean_reward:.3f}")
    print(f"\nVariants ({len(group.variant_runs)}):")

    changes = analyzer.compute_relative_changes(group, metric="final_mean_reward")
    for run in group.variant_runs:
        change = changes[run.name]
        print(f"  {run.name}: {run.final_mean_reward:.3f} ({change:+.1f}%)")

    # Find best config
    best = analyzer.identify_best_configuration(
        runs_training_comparison,
        metric="final_mean_reward",
        prefer_stable=True,
    )
    print(f"\nBest configuration: {best.name}")
    print(f"  Reward: {best.final_mean_reward:.3f} ± {best.final_std_reward:.3f}")

    print("\n" + "=" * 70)
    print("HELPER FUNCTIONS DEMO")
    print("=" * 70 + "\n")

    # Demonstrate helper functions
    print("format_mean_std(0.52, 0.03):")
    print(f"  {generator.format_mean_std(0.52, 0.03)}")

    print("\nformat_delta(0.45, 0.52):")
    print(f"  {generator.format_delta(0.45, 0.52)}")

    print("\nformat_delta(0.52, 0.48):")
    print(f"  {generator.format_delta(0.52, 0.48)}")

    print("\nbold_best_values([0.45, 0.52, 0.48]):")
    print(f"  {generator.bold_best_values([0.45, 0.52, 0.48])}")

    print("\nescape_latex('Train_accuracy: 95% ($\\\\alpha=0.05$)'):")
    print(f"  {generator.escape_latex('Train_accuracy: 95% ($alpha=0.05$)')}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Example LaTeX table generation for NeurIPS paper"
    )
    parser.add_argument(
        "--mode",
        choices=["real", "mock", "both"],
        default="mock",
        help="Run with real data, mock data, or both",
    )

    args = parser.parse_args()

    if args.mode in ["mock", "both"]:
        example_with_mock_data()

    if args.mode in ["real", "both"]:
        print("\n\n")
        example_with_real_data()


if __name__ == "__main__":
    main()
