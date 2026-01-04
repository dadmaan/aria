"""
Test script for benchmark discovery system.

This script demonstrates the usage of the benchmark discovery architecture
and validates it against known directory structures.
"""

from pathlib import Path
from run_discovery import (
    BenchmarkDiscovery,
    StructurePattern,
    MetadataExtractor,
    RunInfo
)


def test_metadata_extraction():
    """Test metadata extraction from directory names."""
    print("=" * 80)
    print("Testing Metadata Extraction")
    print("=" * 80)

    extractor = MetadataExtractor()

    # Test cases
    test_cases = [
        "bench_p1_drqn_h128_s42_251205_1547",
        "bench_p6_obs_centraw_s42_251206_1333",
        "run_drqn_20251207_195352",
        "seed_123"
    ]

    for dirname in test_cases:
        print(f"\nDirectory: {dirname}")

        # Try benchmark extraction
        bench_meta = extractor.extract_benchmark_metadata(dirname)
        if bench_meta:
            print(f"  Benchmark metadata: {bench_meta}")

        # Try run extraction
        run_meta = extractor.extract_run_metadata(dirname)
        if run_meta:
            print(f"  Run metadata: {run_meta}")

        # Try seed extraction
        seed = extractor.extract_seed_from_dirname(dirname)
        if seed:
            print(f"  Seed: {seed}")


def test_structure_detection():
    """Test automatic structure detection."""
    print("\n" + "=" * 80)
    print("Testing Structure Detection")
    print("=" * 80)

    discovery = BenchmarkDiscovery(verbose=True)

    # Test directories (if they exist)
    test_dirs = [
        "/workspace/artifacts/benchmark/20251206_initial",
        "/workspace/artifacts/benchmark/20251207_stateless_lstm",
        "/workspace/artifacts/training"
    ]

    for test_dir in test_dirs:
        path = Path(test_dir)
        if not path.exists():
            print(f"\nSkipping {test_dir} (does not exist)")
            continue

        print(f"\n\nTesting: {test_dir}")
        try:
            pattern = discovery.detect_structure(path)
            print(f"  Detected pattern: {pattern.value}")
        except Exception as e:
            print(f"  Error: {e}")


def test_run_discovery():
    """Test run discovery for each pattern."""
    print("\n" + "=" * 80)
    print("Testing Run Discovery")
    print("=" * 80)

    discovery = BenchmarkDiscovery(verbose=True)

    # Test benchmark directory
    benchmark_dir = Path("/workspace/artifacts/benchmark")
    if benchmark_dir.exists():
        print(f"\n\nDiscovering runs in: {benchmark_dir}")

        # Get all date-based subdirectories
        for date_dir in sorted(benchmark_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            print(f"\n  Checking: {date_dir.name}")

            try:
                pattern = discovery.detect_structure(date_dir)
                print(f"    Pattern: {pattern.value}")

                runs = discovery.discover_runs(date_dir, recursive=True)
                print(f"    Found {len(runs)} runs")

                # Show first few runs
                for i, run in enumerate(runs[:3]):
                    print(f"\n    Run {i+1}:")
                    print(f"      Name: {run.name}")
                    print(f"      Phase: {run.phase}")
                    print(f"      Method: {run.method}")
                    print(f"      Seed: {run.seed}")
                    print(f"      Has config: {run.has_config()}")
                    print(f"      Has metrics: {run.has_metrics()}")
                    print(f"      Has tensorboard: {run.has_tensorboard()}")

                if len(runs) > 3:
                    print(f"\n    ... and {len(runs) - 3} more runs")

            except Exception as e:
                print(f"    Error: {e}")


def test_filtering_and_grouping():
    """Test filtering and grouping functionality."""
    print("\n" + "=" * 80)
    print("Testing Filtering and Grouping")
    print("=" * 80)

    discovery = BenchmarkDiscovery(verbose=False)

    # Use a known directory with runs
    test_dir = Path("/workspace/artifacts/benchmark/20251206_initial")
    if not test_dir.exists():
        print("\nTest directory not found, skipping")
        return

    try:
        runs = discovery.discover_runs(test_dir, recursive=True)
        print(f"\nTotal runs: {len(runs)}")

        # Test filtering
        print("\n--- Filtering Tests ---")

        metrics_runs = discovery.filter_runs(runs, has_metrics=True)
        print(f"Runs with metrics: {len(metrics_runs)}")

        tb_runs = discovery.filter_runs(runs, has_tensorboard=True)
        print(f"Runs with tensorboard: {len(tb_runs)}")

        config_runs = discovery.filter_runs(runs, has_config=True)
        print(f"Runs with config: {len(config_runs)}")

        # Test grouping
        print("\n--- Grouping Tests ---")

        by_phase = discovery.group_runs_by(runs, "phase")
        print(f"\nRuns by phase:")
        for phase, phase_runs in sorted(by_phase.items()):
            print(f"  {phase}: {len(phase_runs)} runs")

        by_method = discovery.group_runs_by(runs, "method")
        print(f"\nRuns by method:")
        for method, method_runs in sorted(by_method.items()):
            print(f"  {method}: {len(method_runs)} runs")

        by_pattern = discovery.group_runs_by(runs, "pattern")
        print(f"\nRuns by pattern:")
        for pattern, pattern_runs in sorted(by_pattern.items(), key=lambda x: x[0].value):
            print(f"  {pattern.value}: {len(pattern_runs)} runs")

    except Exception as e:
        print(f"Error: {e}")


def test_run_info_properties():
    """Test RunInfo properties and methods."""
    print("\n" + "=" * 80)
    print("Testing RunInfo Properties")
    print("=" * 80)

    # Create a sample RunInfo
    run = RunInfo(
        path=Path("/workspace/artifacts/test/run_drqn_20251207_195352"),
        name="run_drqn_20251207_195352",
        phase="phase1",
        method="drqn",
        seed=42,
        variant="h128",
        metadata={"hidden": 128, "timestamp": "20251207_195352"},
        pattern=StructurePattern.ORIGINAL_TRAINING
    )

    print(f"\nRunInfo: {run}")
    print(f"Is valid: {run.is_valid()}")
    print(f"Has config: {run.has_config()}")
    print(f"Has metrics: {run.has_metrics()}")
    print(f"Has tensorboard: {run.has_tensorboard()}")
    print(f"Metadata: {run.metadata}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BENCHMARK DISCOVERY SYSTEM TEST SUITE")
    print("=" * 80)

    test_metadata_extraction()
    test_run_info_properties()
    test_structure_detection()
    test_run_discovery()
    test_filtering_and_grouping()

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
