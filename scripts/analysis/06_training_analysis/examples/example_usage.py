"""
Example Usage of Benchmark Discovery System

This script demonstrates common use cases and patterns for the benchmark
discovery architecture.
"""

from pathlib import Path
from benchmark_discovery import BenchmarkDiscovery, StructurePattern


def example_1_basic_discovery():
    """Example 1: Basic discovery of all runs in a directory."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Discovery")
    print("=" * 80)

    discovery = BenchmarkDiscovery(verbose=True)

    # Discover runs in a benchmark directory
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        runs = discovery.discover_runs(benchmark_dir)

        print(f"\nDiscovered {len(runs)} runs")
        print("\nFirst 3 runs:")
        for i, run in enumerate(runs[:3], 1):
            print(f"\n{i}. {run.name}")
            print(f"   Phase: {run.phase}")
            print(f"   Method: {run.method}")
            print(f"   Seed: {run.seed}")
            print(f"   Path: {run.path}")


def example_2_filter_by_phase():
    """Example 2: Filter runs by phase."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Filter by Phase")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        all_runs = discovery.discover_runs(benchmark_dir)

        # Get only phase1 runs
        phase1_runs = discovery.filter_runs(all_runs, phase="phase1")

        print(f"\nTotal runs: {len(all_runs)}")
        print(f"Phase 1 runs: {len(phase1_runs)}")

        print("\nPhase 1 run details:")
        for run in phase1_runs:
            print(f"  - {run.name} (seed={run.seed})")


def example_3_group_and_analyze():
    """Example 3: Group runs and analyze by category."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Group and Analyze")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        runs = discovery.discover_runs(benchmark_dir)

        # Group by phase
        by_phase = discovery.group_runs_by(runs, "phase")

        print(f"\nRuns grouped by phase:")
        for phase in sorted(by_phase.keys()):
            phase_runs = by_phase[phase]
            print(f"\n{phase}:")
            print(f"  Total: {len(phase_runs)}")

            # Count by method
            methods = {}
            for run in phase_runs:
                method = run.method or "unknown"
                methods[method] = methods.get(method, 0) + 1

            print(f"  Methods: {dict(methods)}")

            # Check data availability
            with_metrics = sum(1 for r in phase_runs if r.has_metrics())
            with_tb = sum(1 for r in phase_runs if r.has_tensorboard())
            print(f"  With metrics: {with_metrics}/{len(phase_runs)}")
            print(f"  With tensorboard: {with_tb}/{len(phase_runs)}")


def example_4_cross_structure_discovery():
    """Example 4: Discover runs across different directory structures."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Cross-Structure Discovery")
    print("=" * 80)

    discovery = BenchmarkDiscovery()

    # Different structure patterns
    test_dirs = [
        "/workspace/artifacts/benchmark/20251206_initial",  # FLAT_BENCHMARK
        "/workspace/artifacts/benchmark/20251207_stateless_lstm",  # METHOD_SEED_HIERARCHY
        "/workspace/artifacts/training",  # ORIGINAL_TRAINING
    ]

    all_runs = []

    for test_dir in test_dirs:
        path = Path(test_dir)
        if not path.exists():
            continue

        try:
            pattern = discovery.detect_structure(path)
            runs = discovery.discover_runs(path)
            all_runs.extend(runs)

            print(f"\n{path.name}:")
            print(f"  Pattern: {pattern.value}")
            print(f"  Runs: {len(runs)}")

        except Exception as e:
            print(f"\n{path.name}: Error - {e}")

    print(f"\n\nTotal runs across all structures: {len(all_runs)}")

    # Group by pattern
    by_pattern = discovery.group_runs_by(all_runs, "pattern")
    print("\nBreakdown by structure pattern:")
    for pattern, pattern_runs in by_pattern.items():
        print(f"  {pattern.value}: {len(pattern_runs)} runs")


def example_5_extract_metadata():
    """Example 5: Extract and use metadata from runs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Metadata Extraction")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        runs = discovery.discover_runs(benchmark_dir)

        # Filter to phase1 for specific analysis
        phase1_runs = discovery.filter_runs(runs, phase="phase1")

        print(f"\nAnalyzing {len(phase1_runs)} Phase 1 runs:")

        for run in phase1_runs:
            print(f"\n{run.name}:")
            print(f"  Method: {run.method}")
            print(f"  Seed: {run.seed}")
            print(f"  Variant: {run.variant}")

            # Access extracted metadata
            if "hidden" in run.metadata:
                print(f"  Hidden size: {run.metadata['hidden']}")
            if "timestamp" in run.metadata:
                print(f"  Timestamp: {run.metadata['timestamp']}")


def example_6_data_availability_check():
    """Example 6: Check data availability across runs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Data Availability Check")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        runs = discovery.discover_runs(benchmark_dir)

        print(f"\nData availability for {len(runs)} runs:")

        # Categorize runs by data availability
        complete_runs = [r for r in runs if r.has_config() and r.has_metrics() and r.has_tensorboard()]
        partial_runs = [r for r in runs if r.is_valid() and r not in complete_runs]
        invalid_runs = [r for r in runs if not r.is_valid()]

        print(f"\nComplete runs (config + metrics + tensorboard): {len(complete_runs)}")
        print(f"Partial runs (some data missing): {len(partial_runs)}")
        print(f"Invalid runs (no data): {len(invalid_runs)}")

        if partial_runs:
            print("\nPartial runs details:")
            for run in partial_runs[:5]:  # Show first 5
                missing = []
                if not run.has_config():
                    missing.append("config")
                if not run.has_metrics():
                    missing.append("metrics")
                if not run.has_tensorboard():
                    missing.append("tensorboard")
                print(f"  {run.name}: missing {', '.join(missing)}")


def example_7_method_seed_hierarchy():
    """Example 7: Work with method/seed hierarchy structure."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Method/Seed Hierarchy")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    hierarchy_dir = Path("/workspace/artifacts/benchmark/20251207_stateless_lstm")

    if hierarchy_dir.exists():
        runs = discovery.discover_runs(hierarchy_dir)

        print(f"\nDiscovered {len(runs)} runs in method/seed hierarchy")

        # Group by seed
        by_seed = discovery.group_runs_by(runs, "seed")

        print(f"\nRuns grouped by seed:")
        for seed in sorted(by_seed.keys()):
            seed_runs = by_seed[seed]
            print(f"\nSeed {seed}:")
            print(f"  Total runs: {len(seed_runs)}")

            # Show methods
            methods = set(r.method for r in seed_runs if r.method)
            print(f"  Methods: {', '.join(sorted(methods))}")

            # Show example run
            if seed_runs:
                example = seed_runs[0]
                print(f"  Example: {example.name}")
                if "method_dir" in example.metadata:
                    print(f"    Method directory: {example.metadata['method_dir']}")


def example_8_combine_filters():
    """Example 8: Combine multiple filters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Combined Filtering")
    print("=" * 80)

    discovery = BenchmarkDiscovery()
    benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")

    if benchmark_dir.exists():
        all_runs = discovery.discover_runs(benchmark_dir)

        # Complex filter: phase1, drqn method, with both metrics and tensorboard
        filtered_runs = discovery.filter_runs(
            all_runs,
            phase="phase1",
            method="drqn",
            has_metrics=True,
            has_tensorboard=True
        )

        print(f"\nTotal runs: {len(all_runs)}")
        print(f"Filtered (phase1 + drqn + metrics + tensorboard): {len(filtered_runs)}")

        print("\nFiltered runs:")
        for run in filtered_runs:
            print(f"  - {run.name}")
            print(f"    Seed: {run.seed}")
            print(f"    Variant: {run.variant}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BENCHMARK DISCOVERY SYSTEM - USAGE EXAMPLES")
    print("=" * 80)

    example_1_basic_discovery()
    example_2_filter_by_phase()
    example_3_group_and_analyze()
    example_4_cross_structure_discovery()
    example_5_extract_metadata()
    example_6_data_availability_check()
    example_7_method_seed_hierarchy()
    example_8_combine_filters()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
