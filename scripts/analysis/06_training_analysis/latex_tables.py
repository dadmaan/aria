#!/usr/bin/env python3
"""
LaTeX Table Generation for NeurIPS Paper

Generates publication-ready LaTeX tables for ablation studies and training comparisons
following NeurIPS 2024 formatting standards.

Usage:
    from latex_tables import LatexTableGenerator, AblationAnalyzer

    generator = LatexTableGenerator()
    table = generator.generate_training_comparison_table(runs, baseline_name="End-to-End DQN")
    print(table)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class TableRow:
    """Represents a single row in a table."""

    name: str
    values: Dict[str, Union[float, str]]
    is_bold: bool = False
    is_baseline: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationGroup:
    """Group of runs sharing an ablation dimension."""

    dimension: str  # e.g., "lr_schedule", "rainbow_component", "curriculum_stage"
    baseline_run: Any  # RunMetrics object
    variant_runs: List[Any]  # List of RunMetrics objects
    aggregated_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # mean, std


class LatexTableGenerator:
    """
    Generates LaTeX tables for NeurIPS paper.

    Supports multiple table formats:
    - Training comparison tables (Table 1 style)
    - Ablation study tables (component removal)
    - Grid search results
    - Learning rate schedule comparisons
    """

    def __init__(
        self,
        decimals: int = 2,
        use_booktabs: bool = True,
        bold_best: bool = True,
        show_delta: bool = True,
    ):
        """
        Initialize the table generator.

        Args:
            decimals: Number of decimal places for formatting
            use_booktabs: Use booktabs package (toprule, midrule, bottomrule)
            bold_best: Automatically bold the best values
            show_delta: Show relative change columns in ablation tables
        """
        self.decimals = decimals
        self.use_booktabs = use_booktabs
        self.bold_best = bold_best
        self.show_delta = show_delta

    def generate_training_comparison_table(
        self,
        runs: List[Any],
        baseline_name: str,
        caption: str = "Training performance comparison. Mean $\\pm$ std over 5 seeds.",
        label: str = "tab:training_results",
        include_metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate training performance comparison table (Table 1 style).

        Args:
            runs: List of RunMetrics objects
            baseline_name: Name of the baseline run for relative comparisons
            caption: Table caption
            label: LaTeX label for cross-referencing
            include_metrics: Metrics to include. Default: ['reward', 'time', 'efficiency']

        Returns:
            Complete LaTeX table as string
        """
        if include_metrics is None:
            include_metrics = ['reward', 'time', 'efficiency']

        # Build table header
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")

        # Column specification
        col_spec = "l" + "c" * len(include_metrics)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        # Header row
        if self.use_booktabs:
            lines.append("\\toprule")
        else:
            lines.append("\\hline")

        header_cols = ["\\textbf{Method}"]
        if 'reward' in include_metrics:
            header_cols.append("\\textbf{Episode Reward}")
        if 'time' in include_metrics:
            header_cols.append("\\textbf{Training Time}")
        if 'efficiency' in include_metrics:
            header_cols.append("\\textbf{Sample Efficiency}")

        lines.append(" & ".join(header_cols) + " \\\\")

        if self.use_booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")

        # Find baseline for relative calculations
        baseline_run = None
        for run in runs:
            if baseline_name.lower() in run.name.lower():
                baseline_run = run
                break

        if baseline_run is None:
            baseline_run = runs[0]  # Fallback to first run

        baseline_time = baseline_run.metadata.get('training_time_hours', 1.0)
        baseline_sample_efficiency = 1.0

        # Sort runs by final mean reward (descending)
        sorted_runs = sorted(runs, key=lambda r: r.final_mean_reward, reverse=True)

        # Data rows
        for run in sorted_runs:
            row_cols = []

            # Method name
            method_name = self._format_method_name(run.name)
            if run == sorted_runs[0] and self.bold_best:
                method_name = f"\\textbf{{{method_name}}}"
            row_cols.append(method_name)

            # Episode reward
            if 'reward' in include_metrics:
                reward_str = self.format_mean_std(
                    run.final_mean_reward,
                    run.final_std_reward,
                    self.decimals
                )
                if run == sorted_runs[0] and self.bold_best:
                    reward_str = f"$\\mathbf{{{reward_str[1:-1]}}}$"
                row_cols.append(reward_str)

            # Training time
            if 'time' in include_metrics:
                time_hours = run.metadata.get('training_time_hours', 0.0)
                if time_hours > 0:
                    time_str = f"{time_hours:.1f}h"
                else:
                    time_str = "[TBD]h"
                row_cols.append(time_str)

            # Sample efficiency (relative to baseline)
            if 'efficiency' in include_metrics:
                time_hours = run.metadata.get('training_time_hours', baseline_time)
                if time_hours > 0 and baseline_time > 0:
                    efficiency = baseline_time / time_hours
                    efficiency_str = f"{efficiency:.1f}$\\times$"
                else:
                    efficiency_str = "[TBD]$\\times$"
                row_cols.append(efficiency_str)

            lines.append(" & ".join(row_cols) + " \\\\")

        # Footer
        if self.use_booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_ablation_table(
        self,
        runs: List[Any],
        full_method_name: str,
        components: List[str],
        caption: str = "Ablation study results.",
        label: str = "tab:ablation",
        use_relative_change: bool = True,
    ) -> str:
        """
        Generate component ablation table.

        Args:
            runs: List of RunMetrics objects
            full_method_name: Name of the run with all components
            components: List of component names in order
            caption: Table caption
            label: LaTeX label
            use_relative_change: Show percentage change instead of absolute

        Returns:
            Complete LaTeX table as string
        """
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")

        # Column specification
        if use_relative_change:
            lines.append("\\begin{tabular}{lcc}")
        else:
            lines.append("\\begin{tabular}{lc}")

        # Header
        if self.use_booktabs:
            lines.append("\\toprule")

        if use_relative_change:
            lines.append("\\textbf{Variant} & \\textbf{Reward} & \\textbf{$\\Delta$} \\\\")
        else:
            lines.append("\\textbf{Variant} & \\textbf{Reward} \\\\")

        if self.use_booktabs:
            lines.append("\\midrule")

        # Find full method run
        full_run = None
        for run in runs:
            if full_method_name.lower() in run.name.lower():
                full_run = run
                break

        if full_run is None:
            # Use best performing as baseline
            full_run = max(runs, key=lambda r: r.final_mean_reward)

        baseline_reward = full_run.final_mean_reward

        # Full method row
        reward_str = f"${baseline_reward:.{self.decimals}f}$"
        if use_relative_change:
            lines.append(f"Full method & {reward_str} & --- \\\\")
        else:
            lines.append(f"Full method & {reward_str} \\\\")

        # Ablation rows (sorted by performance)
        ablation_runs = [r for r in runs if r != full_run]
        ablation_runs.sort(key=lambda r: r.final_mean_reward, reverse=True)

        for run in ablation_runs:
            variant_name = self._format_ablation_variant(run.name, components)
            reward = run.final_mean_reward
            reward_str = f"${reward:.{self.decimals}f}$"

            if use_relative_change:
                delta_str = self.format_delta(baseline_reward, reward)
                lines.append(f"{variant_name} & {reward_str} & {delta_str} \\\\")
            else:
                lines.append(f"{variant_name} & {reward_str} \\\\")

        # Footer
        if self.use_booktabs:
            lines.append("\\bottomrule")

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_lr_schedule_table(
        self,
        runs: List[Any],
        caption: str = "Ablation: learning rate schedule.",
        label: str = "tab:lr_ablation",
    ) -> str:
        """
        Generate learning rate schedule comparison table.

        Args:
            runs: List of RunMetrics objects with different LR schedules
            caption: Table caption
            label: LaTeX label

        Returns:
            Complete LaTeX table as string
        """
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{lccc}")

        if self.use_booktabs:
            lines.append("\\toprule")

        lines.append("\\textbf{LR Schedule} & \\textbf{Initial LR} & \\textbf{Reward} & \\textbf{$\\Delta$} \\\\")

        if self.use_booktabs:
            lines.append("\\midrule")

        # Sort by performance
        sorted_runs = sorted(runs, key=lambda r: r.final_mean_reward, reverse=True)
        best_reward = sorted_runs[0].final_mean_reward

        # Identify schedule types from config
        for i, run in enumerate(sorted_runs):
            config = run.config
            training = config.get('training', {})
            lr_sched = training.get('learning_rate_scheduler', {})

            # Determine schedule name
            if lr_sched.get('enabled'):
                sched_type = lr_sched.get('type', 'unknown')
                initial_lr = lr_sched.get('initial_lr', training.get('learning_rate', 0.001))
                final_lr = lr_sched.get('final_lr', initial_lr)

                if sched_type == 'exponential':
                    schedule_name = "Exponential decay"
                    lr_display = f"${initial_lr:.0e} \\to {final_lr:.0e}$"
                elif sched_type == 'linear':
                    schedule_name = "Linear decay"
                    lr_display = f"${initial_lr:.0e} \\to {final_lr:.0e}$"
                else:
                    schedule_name = sched_type.capitalize()
                    lr_display = f"${initial_lr:.0e} \\to {final_lr:.0e}$"
            else:
                # Constant LR
                lr_value = training.get('learning_rate', 0.001)
                schedule_name = f"Constant (${lr_value:.0e}$)"
                lr_display = f"${lr_value:.0e}$"

            # Make best entry bold
            if i == 0 and self.bold_best:
                schedule_name = f"\\textbf{{{schedule_name}}}"
                lr_display = f"\\textbf{{{lr_display}}}"

            # Reward
            reward = run.final_mean_reward
            reward_str = f"${reward:.{self.decimals}f}$"
            if i == 0 and self.bold_best:
                reward_str = f"$\\mathbf{{{reward:.{self.decimals}f}}}$"

            # Delta (relative to first constant LR run if this is baseline)
            if i == len(sorted_runs) - 1:
                # Last row is baseline
                delta_str = "---"
            else:
                baseline_reward = sorted_runs[-1].final_mean_reward
                delta_str = self.format_delta(baseline_reward, reward)
                if i == 0 and self.bold_best:
                    # Extract percentage and make bold
                    delta_value = ((reward - baseline_reward) / abs(baseline_reward)) * 100
                    delta_str = f"$\\mathbf{{{delta_value:+.1f}\\%}}$"

            lines.append(f"{schedule_name} & {lr_display} & {reward_str} & {delta_str} \\\\")

        if self.use_booktabs:
            lines.append("\\bottomrule")

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def generate_hyperparameter_grid_table(
        self,
        runs: List[Any],
        param_name: str,
        param_display_name: str,
        caption: str = "Grid search results.",
        label: str = "tab:grid_search",
        additional_columns: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate hyperparameter grid search results table.

        Args:
            runs: List of RunMetrics objects
            param_name: Parameter name in config (e.g., 'diversity_range')
            param_display_name: Display name for table header
            caption: Table caption
            label: LaTeX label
            additional_columns: Dict mapping column names to config paths

        Returns:
            Complete LaTeX table as string
        """
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")

        # Column spec
        n_cols = 2  # param + reward
        if additional_columns:
            n_cols += len(additional_columns)
        col_spec = "l" + "c" * (n_cols - 1)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

        if self.use_booktabs:
            lines.append("\\toprule")

        # Header
        header_cols = [f"\\textbf{{{param_display_name}}}"]
        if additional_columns:
            for col_name in additional_columns.keys():
                header_cols.append(f"\\textbf{{{col_name}}}")
        header_cols.append("\\textbf{Reward}")

        lines.append(" & ".join(header_cols) + " \\\\")

        if self.use_booktabs:
            lines.append("\\midrule")

        # Sort by reward
        sorted_runs = sorted(runs, key=lambda r: r.final_mean_reward, reverse=True)
        best_reward = sorted_runs[0].final_mean_reward

        # Data rows
        for i, run in enumerate(sorted_runs):
            row_cols = []

            # Parameter value
            param_value = self._extract_param_value(run.config, param_name)
            param_str = self._format_param_value(param_value)

            # Make best row bold
            if i == 0 and self.bold_best:
                param_str = f"$\\mathbf{{{param_str[1:-1]}}}$"

            row_cols.append(param_str)

            # Additional columns
            if additional_columns:
                for col_path in additional_columns.values():
                    col_value = self._extract_param_value(run.config, col_path)
                    col_str = self._format_param_value(col_value)
                    if i == 0 and self.bold_best:
                        col_str = f"\\textbf{{{col_str}}}"
                    row_cols.append(col_str)

            # Reward
            reward = run.final_mean_reward
            reward_str = f"${reward:.{self.decimals}f}$"
            if i == 0 and self.bold_best:
                reward_str = f"$\\mathbf{{{reward:.{self.decimals}f}}}$"

            row_cols.append(reward_str)

            lines.append(" & ".join(row_cols) + " \\\\")

        if self.use_booktabs:
            lines.append("\\bottomrule")

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # ==================== Helper Functions ====================

    @staticmethod
    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
        """
        Format mean ± std for LaTeX.

        Args:
            mean: Mean value
            std: Standard deviation
            decimals: Number of decimal places

        Returns:
            Formatted string like "$0.52 \\pm 0.03$"
        """
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    @staticmethod
    def format_delta(baseline: float, current: float, decimals: int = 1) -> str:
        """
        Format relative change as percentage.

        Args:
            baseline: Baseline value
            current: Current value
            decimals: Number of decimal places

        Returns:
            Formatted string like "$+5.2\\%$" or "$-3.1\\%$"
        """
        if baseline == 0:
            return "---"

        delta_pct = ((current - baseline) / abs(baseline)) * 100
        sign = "+" if delta_pct >= 0 else ""
        return f"${sign}{delta_pct:.{decimals}f}\\%$"

    @staticmethod
    def bold_best_values(values: List[float], format_str: str = "{:.2f}") -> List[str]:
        """
        Format values with best value in bold.

        Args:
            values: List of numeric values
            format_str: Format string for numbers

        Returns:
            List of formatted strings with max value bolded
        """
        if not values:
            return []

        max_val = max(values)
        formatted = []

        for val in values:
            val_str = format_str.format(val)
            if val == max_val:
                formatted.append(f"$\\mathbf{{{val_str}}}$")
            else:
                formatted.append(f"${val_str}$")

        return formatted

    @staticmethod
    def escape_latex(text: str) -> str:
        """
        Escape special LaTeX characters.

        Args:
            text: Input text

        Returns:
            Text with escaped special characters
        """
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
            '\\': '\\textbackslash{}',
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def _format_method_name(self, name: str) -> str:
        """Format method name for display."""
        # Remove 'run_' prefix
        name = name.replace('run_', '')

        # Replace common patterns
        replacements = {
            'rainbow-drqn': 'Rainbow DRQN',
            'vanilla-drqn': 'Vanilla DRQN',
            'end-to-end': 'End-to-End',
            'single-agent': 'Single-Agent',
            'joint-training': 'Joint Training',
            'no-curriculum': 'No Curriculum',
            '-': ' ',
            '_': ' ',
        }

        for old, new in replacements.items():
            name = name.replace(old, new)

        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())

        return name

    def _format_ablation_variant(self, name: str, components: List[str]) -> str:
        """Format ablation variant name."""
        name = name.replace('run_', '').replace('_', ' ').replace('-', ' ')

        # Check if it's a removal variant
        for component in components:
            component_clean = component.replace('_', ' ').replace('-', ' ')
            if f"no {component_clean}" in name.lower() or f"without {component_clean}" in name.lower():
                return f"$-$ {component.replace('_', ' ').capitalize()}"

        # Otherwise just clean up the name
        return name.capitalize()

    def _extract_param_value(self, config: Dict[str, Any], param_path: str) -> Any:
        """Extract parameter value from nested config using dot notation."""
        keys = param_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value

    def _format_param_value(self, value: Any) -> str:
        """Format parameter value for display."""
        if value is None:
            return "[TBD]"

        if isinstance(value, (list, tuple)):
            # Format ranges like [0.62, 0.75]
            if len(value) == 2:
                return f"$[{value[0]:.2f}, {value[1]:.2f}]$"
            else:
                return f"${value}$"

        if isinstance(value, float):
            return f"${value:.2f}$"

        if isinstance(value, bool):
            return "Yes" if value else "No"

        return str(value)


class AblationAnalyzer:
    """
    Analyzes ablation studies and groups runs by dimension.

    Helps organize runs for generating ablation tables by identifying
    what was varied across different runs.
    """

    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize analyzer.

        Args:
            significance_threshold: P-value threshold for statistical tests
        """
        self.significance_threshold = significance_threshold

    def group_by_dimension(
        self,
        runs: List[Any],
        dimension: str,
        baseline_value: Any = None,
    ) -> AblationGroup:
        """
        Group runs by ablation dimension.

        Args:
            runs: List of RunMetrics objects
            dimension: Config path to the ablation dimension (e.g., 'training.learning_rate')
            baseline_value: Value that identifies the baseline run

        Returns:
            AblationGroup with organized runs
        """
        # Extract dimension values for each run
        run_values = []
        for run in runs:
            value = self._extract_nested_value(run.config, dimension)
            run_values.append((run, value))

        # Identify baseline
        baseline_run = None
        variant_runs = []

        if baseline_value is not None:
            for run, value in run_values:
                if value == baseline_value:
                    baseline_run = run
                else:
                    variant_runs.append(run)
        else:
            # Use best performing as baseline
            baseline_run = max(runs, key=lambda r: r.final_mean_reward)
            variant_runs = [r for r in runs if r != baseline_run]

        group = AblationGroup(
            dimension=dimension,
            baseline_run=baseline_run,
            variant_runs=variant_runs,
        )

        return group

    def compute_relative_changes(
        self,
        group: AblationGroup,
        metric: str = 'final_mean_reward',
    ) -> Dict[str, float]:
        """
        Compute relative changes from baseline for each variant.

        Args:
            group: AblationGroup to analyze
            metric: Metric to compare (attribute name in RunMetrics)

        Returns:
            Dict mapping run names to relative changes (percentage)
        """
        baseline_value = getattr(group.baseline_run, metric)
        changes = {}

        for run in group.variant_runs:
            variant_value = getattr(run, metric)
            if baseline_value != 0:
                pct_change = ((variant_value - baseline_value) / abs(baseline_value)) * 100
                changes[run.name] = pct_change
            else:
                changes[run.name] = 0.0

        return changes

    def test_significance(
        self,
        run1: Any,
        run2: Any,
        use_final_phase: bool = True,
        test: str = 'mannwhitneyu',
    ) -> Tuple[float, bool]:
        """
        Test statistical significance between two runs.

        Args:
            run1: First RunMetrics object
            run2: Second RunMetrics object
            use_final_phase: Use only final 20% of episodes
            test: Statistical test to use ('mannwhitneyu' or 't-test')

        Returns:
            Tuple of (p_value, is_significant)
        """
        # Extract rewards
        if use_final_phase:
            late_start1 = int(len(run1.episode_rewards) * 0.8)
            late_start2 = int(len(run2.episode_rewards) * 0.8)
            rewards1 = run1.episode_rewards[late_start1:]
            rewards2 = run2.episode_rewards[late_start2:]
        else:
            rewards1 = run1.episode_rewards
            rewards2 = run2.episode_rewards

        # Perform test
        if test == 'mannwhitneyu':
            statistic, p_value = stats.mannwhitneyu(
                rewards1, rewards2, alternative='two-sided'
            )
        elif test == 't-test':
            statistic, p_value = stats.ttest_ind(rewards1, rewards2)
        else:
            raise ValueError(f"Unknown test: {test}")

        is_significant = p_value < self.significance_threshold

        return p_value, is_significant

    def aggregate_across_seeds(
        self,
        runs: List[Any],
        group_by: str = 'config_hash',
    ) -> Dict[str, Tuple[float, float]]:
        """
        Aggregate metrics across multiple seeds.

        Args:
            runs: List of RunMetrics objects (multiple seeds of same config)
            group_by: How to group runs (e.g., 'config_hash' or config path)

        Returns:
            Dict mapping group names to (mean, std) tuples
        """
        # Group runs
        groups = {}
        for run in runs:
            if group_by == 'config_hash':
                # Use first part of name as group identifier
                group_name = run.name.split('_seed')[0] if '_seed' in run.name else run.name
            else:
                # Use config value
                group_name = str(self._extract_nested_value(run.config, group_by))

            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(run.final_mean_reward)

        # Compute statistics
        aggregated = {}
        for group_name, rewards in groups.items():
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            aggregated[group_name] = (mean_reward, std_reward)

        return aggregated

    def identify_best_configuration(
        self,
        runs: List[Any],
        metric: str = 'final_mean_reward',
        prefer_stable: bool = False,
    ) -> Any:
        """
        Identify the best configuration from a set of runs.

        Args:
            runs: List of RunMetrics objects
            metric: Primary metric for comparison
            prefer_stable: Prefer more stable runs when performance is close

        Returns:
            Best RunMetrics object
        """
        if not runs:
            return None

        if prefer_stable:
            # Score = metric - penalty for instability
            scores = []
            for run in runs:
                metric_value = getattr(run, metric)
                stability_penalty = (1 - run.stability_score) * 0.1 * metric_value
                score = metric_value - stability_penalty
                scores.append(score)

            best_idx = np.argmax(scores)
            return runs[best_idx]
        else:
            # Simply use metric
            return max(runs, key=lambda r: getattr(r, metric))

    def _extract_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Extract value from nested dict using dot notation."""
        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value


# ==================== Example Usage ====================

def example_usage():
    """Example demonstrating how to use the table generators."""
    from dataclasses import dataclass, field
    import numpy as np

    # Mock RunMetrics for demonstration
    @dataclass
    class MockRunMetrics:
        name: str
        final_mean_reward: float
        final_std_reward: float
        episode_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
        stability_score: float = 0.8
        config: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)

    # Create sample runs
    runs = [
        MockRunMetrics(
            name="run_end-to-end-dqn",
            final_mean_reward=0.45,
            final_std_reward=0.03,
            config={'training': {'learning_rate': 0.001}},
            metadata={'training_time_hours': 2.1},
        ),
        MockRunMetrics(
            name="run_rainbow-drqn",
            final_mean_reward=0.52,
            final_std_reward=0.02,
            config={'training': {'learning_rate': 0.001}},
            metadata={'training_time_hours': 1.8},
        ),
        MockRunMetrics(
            name="run_vanilla-drqn",
            final_mean_reward=0.48,
            final_std_reward=0.04,
            config={'training': {'learning_rate': 0.001}},
            metadata={'training_time_hours': 1.9},
        ),
    ]

    # Generate tables
    generator = LatexTableGenerator()

    # Training comparison
    table1 = generator.generate_training_comparison_table(
        runs=runs,
        baseline_name="end-to-end-dqn",
    )
    print("=" * 60)
    print("TRAINING COMPARISON TABLE")
    print("=" * 60)
    print(table1)
    print()

    # Ablation study
    ablation_runs = [
        MockRunMetrics(
            name="run_full-rainbow",
            final_mean_reward=0.52,
            final_std_reward=0.02,
        ),
        MockRunMetrics(
            name="run_no-c51",
            final_mean_reward=0.48,
            final_std_reward=0.03,
        ),
        MockRunMetrics(
            name="run_no-prioritized-replay",
            final_mean_reward=0.49,
            final_std_reward=0.02,
        ),
    ]

    table2 = generator.generate_ablation_table(
        runs=ablation_runs,
        full_method_name="full-rainbow",
        components=["c51", "prioritized-replay"],
        caption="Ablation: Rainbow DRQN components.",
        label="tab:rainbow_ablation",
    )
    print("=" * 60)
    print("ABLATION TABLE")
    print("=" * 60)
    print(table2)


def generate_all_tables_from_runs(
    runs: List[Any],
    output_dir: Path,
    baseline_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate all standard tables from a list of runs.

    Args:
        runs: List of RunMetrics objects
        output_dir: Directory to save LaTeX files
        baseline_name: Name of baseline run (auto-detected if None)

    Returns:
        Dict mapping table names to LaTeX code
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = LatexTableGenerator()
    tables = {}

    # Auto-detect baseline
    if baseline_name is None:
        # Look for common baseline patterns
        for run in runs:
            if any(pattern in run.name.lower() for pattern in ['baseline', 'vanilla', 'end-to-end']):
                baseline_name = run.name
                break

        if baseline_name is None:
            baseline_name = runs[0].name

    # Training comparison
    tables['training_comparison'] = generator.generate_training_comparison_table(
        runs=runs,
        baseline_name=baseline_name,
    )

    # Save to file
    for name, latex_code in tables.items():
        output_file = output_dir / f"{name}.tex"
        with open(output_file, 'w') as f:
            f.write(latex_code)

    return tables


if __name__ == "__main__":
    example_usage()
