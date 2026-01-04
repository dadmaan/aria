"""
Integration Example: Benchmark Discovery + Data Loaders

This script demonstrates how to integrate the benchmark discovery system
with existing data loading and analysis tools.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd

from benchmark_discovery import BenchmarkDiscovery, RunInfo, StructurePattern


class IntegratedAnalyzer:
    """
    Analyzer that combines benchmark discovery with data loading capabilities.

    This class demonstrates how to build analysis pipelines on top of the
    discovery architecture.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the integrated analyzer.

        Args:
            verbose: Enable verbose logging
        """
        self.discovery = BenchmarkDiscovery(verbose=verbose)
        self.verbose = verbose

    def load_metrics_from_run(self, run: RunInfo) -> pd.DataFrame:
        """
        Load training metrics from a run into a DataFrame.

        Args:
            run: RunInfo object with metrics

        Returns:
            DataFrame with training metrics
        """
        if not run.has_metrics():
            return pd.DataFrame()

        with open(run.metrics_path) as f:
            metrics = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Add metadata columns
        df['run_name'] = run.name
        df['phase'] = run.phase
        df['method'] = run.method
        df['seed'] = run.seed
        df['variant'] = run.variant

        return df

    def load_all_metrics(self, runs: List[RunInfo]) -> pd.DataFrame:
        """
        Load metrics from all runs into a single DataFrame.

        Args:
            runs: List of RunInfo objects

        Returns:
            Combined DataFrame with all metrics
        """
        dfs = []

        for run in runs:
            if run.has_metrics():
                df = self.load_metrics_from_run(run)
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def analyze_phase(self, benchmark_dir: Path, phase: str) -> Dict[str, Any]:
        """
        Analyze all runs from a specific phase.

        Args:
            benchmark_dir: Root benchmark directory
            phase: Phase identifier (e.g., "phase1")

        Returns:
            Dictionary with analysis results
        """
        # Discover runs
        all_runs = self.discovery.discover_runs(benchmark_dir)
        phase_runs = self.discovery.filter_runs(all_runs, phase=phase, has_metrics=True)

        if not phase_runs:
            return {"error": f"No runs found for {phase}"}

        # Load metrics
        df = self.load_all_metrics(phase_runs)

        # Compute statistics
        results = {
            "phase": phase,
            "total_runs": len(phase_runs),
            "methods": list(df['method'].unique()) if 'method' in df.columns else [],
            "seeds": sorted(df['seed'].unique().tolist()) if 'seed' in df.columns else [],
        }

        # Per-method statistics
        if 'method' in df.columns and 'reward' in df.columns:
            method_stats = df.groupby('method')['reward'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).to_dict('index')
            results["method_statistics"] = method_stats

        # Per-seed statistics
        if 'seed' in df.columns and 'reward' in df.columns:
            seed_stats = df.groupby('seed')['reward'].agg([
                'count', 'mean', 'std'
            ]).to_dict('index')
            results["seed_statistics"] = seed_stats

        return results

    def compare_methods(
        self,
        benchmark_dir: Path,
        methods: List[str],
        phase: str = None
    ) -> pd.DataFrame:
        """
        Compare multiple methods across runs.

        Args:
            benchmark_dir: Root benchmark directory
            methods: List of method names to compare
            phase: Optional phase filter

        Returns:
            DataFrame with comparison statistics
        """
        # Discover runs
        all_runs = self.discovery.discover_runs(benchmark_dir)

        # Filter by phase if specified
        if phase:
            all_runs = self.discovery.filter_runs(all_runs, phase=phase)

        # Filter by methods
        comparison_runs = []
        for method in methods:
            method_runs = self.discovery.filter_runs(
                all_runs,
                method=method,
                has_metrics=True
            )
            comparison_runs.extend(method_runs)

        if not comparison_runs:
            return pd.DataFrame()

        # Load metrics
        df = self.load_all_metrics(comparison_runs)

        # Compute comparison statistics
        if 'reward' in df.columns:
            comparison = df.groupby(['method', 'seed']).agg({
                'reward': ['count', 'mean', 'std', 'max'],
                'episode': 'max'
            }).reset_index()

            return comparison

        return pd.DataFrame()

    def discover_and_summarize(self, root_dir: Path) -> Dict[str, Any]:
        """
        Discover all runs under a root directory and provide a summary.

        Args:
            root_dir: Root directory to search

        Returns:
            Summary dictionary
        """
        # Detect structure
        try:
            pattern = self.discovery.detect_structure(root_dir)
        except Exception as e:
            return {"error": str(e)}

        # Discover runs
        runs = self.discovery.discover_runs(root_dir)

        # Compute summary statistics
        summary = {
            "directory": str(root_dir),
            "pattern": pattern.value,
            "total_runs": len(runs),
            "valid_runs": sum(1 for r in runs if r.is_valid()),
            "with_metrics": sum(1 for r in runs if r.has_metrics()),
            "with_tensorboard": sum(1 for r in runs if r.has_tensorboard()),
            "with_config": sum(1 for r in runs if r.has_config()),
        }

        # Group by various attributes
        by_phase = self.discovery.group_runs_by(runs, "phase")
        summary["phases"] = {
            phase: len(phase_runs) for phase, phase_runs in by_phase.items()
        }

        by_method = self.discovery.group_runs_by(runs, "method")
        summary["methods"] = {
            method: len(method_runs) for method, method_runs in by_method.items()
        }

        by_seed = self.discovery.group_runs_by(runs, "seed")
        summary["seeds"] = {
            seed: len(seed_runs) for seed, seed_runs in by_seed.items()
        }

        return summary

    def export_run_catalog(self, root_dir: Path, output_file: Path):
        """
        Export a catalog of all discovered runs to CSV.

        Args:
            root_dir: Root directory to search
            output_file: Output CSV file path
        """
        # Discover runs
        runs = self.discovery.discover_runs(root_dir)

        # Convert to DataFrame
        data = []
        for run in runs:
            row = {
                "name": run.name,
                "path": str(run.path),
                "phase": run.phase,
                "method": run.method,
                "seed": run.seed,
                "variant": run.variant,
                "pattern": run.pattern.value if run.pattern else None,
                "has_config": run.has_config(),
                "has_metrics": run.has_metrics(),
                "has_tensorboard": run.has_tensorboard(),
            }

            # Add metadata fields
            for key, value in run.metadata.items():
                if key not in row:
                    row[f"meta_{key}"] = value

            data.append(row)

        df = pd.DataFrame(data)

        # Export to CSV
        df.to_csv(output_file, index=False)

        if self.verbose:
            print(f"Exported {len(runs)} runs to {output_file}")

        return df


def example_1_integrated_analysis():
    """Example 1: Integrated discovery and analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Integrated Discovery and Analysis")
    print("=" * 80)

    analyzer = IntegratedAnalyzer(verbose=True)

    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        # Analyze phase1
        results = analyzer.analyze_phase(benchmark_dir, "phase1")

        print("\nPhase 1 Analysis Results:")
        print(f"  Total runs: {results.get('total_runs')}")
        print(f"  Methods: {results.get('methods')}")
        print(f"  Seeds: {results.get('seeds')}")

        if "method_statistics" in results:
            print("\n  Method Statistics (reward):")
            for method, stats in results["method_statistics"].items():
                print(f"    {method}:")
                print(f"      Count: {stats['count']}")
                print(f"      Mean: {stats['mean']:.2f}")
                print(f"      Std: {stats['std']:.2f}")


def example_2_method_comparison():
    """Example 2: Compare multiple methods."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Method Comparison")
    print("=" * 80)

    analyzer = IntegratedAnalyzer()

    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        # Compare DRQN and MLP in phase1
        comparison = analyzer.compare_methods(
            benchmark_dir,
            methods=["drqn", "mlp"],
            phase="phase1"
        )

        if not comparison.empty:
            print("\nMethod Comparison (Phase 1):")
            print(comparison.to_string())


def example_3_summary_report():
    """Example 3: Generate summary report."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Summary Report")
    print("=" * 80)

    analyzer = IntegratedAnalyzer()

    test_dirs = [
        "/workspace/artifacts/benchmark/20251206_initial",
        "/workspace/artifacts/benchmark/20251207_stateless_lstm",
    ]

    for test_dir in test_dirs:
        path = Path(test_dir)
        if not path.exists():
            continue

        summary = analyzer.discover_and_summarize(path)

        print(f"\n{path.name}:")
        print(f"  Pattern: {summary.get('pattern')}")
        print(f"  Total runs: {summary.get('total_runs')}")
        print(f"  Valid runs: {summary.get('valid_runs')}")
        print(f"  With metrics: {summary.get('with_metrics')}")
        print(f"  With tensorboard: {summary.get('with_tensorboard')}")

        if "phases" in summary and summary["phases"]:
            print(f"\n  Phases:")
            for phase, count in sorted(summary["phases"].items()):
                print(f"    {phase}: {count} runs")

        if "methods" in summary and summary["methods"]:
            print(f"\n  Methods:")
            for method, count in sorted(summary["methods"].items()):
                print(f"    {method}: {count} runs")


def example_4_export_catalog():
    """Example 4: Export run catalog to CSV."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Export Run Catalog")
    print("=" * 80)

    analyzer = IntegratedAnalyzer(verbose=True)

    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")
    output_file = Path("/tmp/run_catalog.csv")

    if benchmark_dir.exists():
        df = analyzer.export_run_catalog(benchmark_dir, output_file)

        print(f"\nExported catalog with {len(df)} runs")
        print(f"\nFirst few rows:")
        print(df[['name', 'phase', 'method', 'seed', 'has_metrics']].head(10))


def example_5_load_and_aggregate():
    """Example 5: Load and aggregate metrics across runs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Load and Aggregate Metrics")
    print("=" * 80)

    analyzer = IntegratedAnalyzer()
    discovery = BenchmarkDiscovery()

    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        # Get all phase1 runs
        all_runs = discovery.discover_runs(benchmark_dir)
        phase1_runs = discovery.filter_runs(all_runs, phase="phase1", has_metrics=True)

        # Load all metrics
        df = analyzer.load_all_metrics(phase1_runs)

        if not df.empty:
            print(f"\nLoaded {len(df)} episodes from {len(phase1_runs)} runs")
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")

            # Aggregate statistics
            if 'reward' in df.columns:
                print(f"\nOverall reward statistics:")
                print(f"  Mean: {df['reward'].mean():.2f}")
                print(f"  Std: {df['reward'].std():.2f}")
                print(f"  Min: {df['reward'].min():.2f}")
                print(f"  Max: {df['reward'].max():.2f}")

            # Group by method
            if 'method' in df.columns and 'reward' in df.columns:
                print(f"\nReward by method:")
                method_means = df.groupby('method')['reward'].mean()
                for method, mean_reward in method_means.items():
                    print(f"  {method}: {mean_reward:.2f}")


def main():
    """Run all integration examples."""
    print("\n" + "=" * 80)
    print("BENCHMARK DISCOVERY + DATA LOADING INTEGRATION")
    print("=" * 80)

    example_1_integrated_analysis()
    example_2_method_comparison()
    example_3_summary_report()
    example_4_export_catalog()
    example_5_load_and_aggregate()

    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
