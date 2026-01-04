#!/usr/bin/env python3
"""Complete Inference Pipeline with Human-in-the-Loop Support.

This script provides a unified CLI for running the complete inference
and evaluation pipeline including:
    - Batch sequence generation
    - Interactive HIL sessions
    - Sequence analysis
    - Benchmark comparisons
    - Paper results generation

Usage:
    # Run interactive HIL session
    python scripts/inference/run_inference_pipeline.py hil \
        --checkpoint artifacts/training/.../final.pth \
        --num-iterations 10

    # Analyze existing sequences
    python scripts/inference/run_inference_pipeline.py analyze \
        --sequences outputs/inference_cl_benchmark/drqn/sequences.json

    # Run benchmark comparison
    python scripts/inference/run_inference_pipeline.py benchmark \
        --checkpoints-dir artifacts/benchmark/20251210_cl_benchmark

    # Generate paper results
    python scripts/inference/run_inference_pipeline.py paper \
        --results-dir outputs/benchmark \
        --output nips_paper/figures/generated
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging.logging_manager import get_logger
from src.inference.config_loader import InferenceConfig

logger = get_logger("inference_pipeline")
console = Console()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# HIL Session Command
# ============================================================================


def cmd_hil(args: argparse.Namespace) -> int:
    """Run interactive Human-in-the-Loop session."""
    from scripts.inference.run_inference import (
        setup_environment,
        load_agent,
        load_checkpoint_config,
        detect_action_space_from_checkpoint,
    )
    from src.inference.interactive_session import InteractiveInferenceSession

    console.print(
        Panel(
            "[bold cyan]Human-in-the-Loop Inference Session[/bold cyan]",
            subtitle="Phase 2 Implementation",
        )
    )

    # Load configuration
    config = load_config(args.config)

    # Merge checkpoint config
    checkpoint_config = load_checkpoint_config(args.checkpoint)
    if checkpoint_config:
        if "network" in checkpoint_config:
            config["network"] = checkpoint_config["network"]
        if "algorithm" in checkpoint_config:
            config["algorithm"] = checkpoint_config["algorithm"]

    # Setup environment
    action_space_size = detect_action_space_from_checkpoint(args.checkpoint)
    env, ghsom_manager = setup_environment(config, action_space_size)

    # Load agent
    trainer = load_agent(args.checkpoint, env, config)

    # Create interactive session
    session = InteractiveInferenceSession(
        trainer=trainer,
        env=env,
        ghsom_manager=ghsom_manager,
        config=config,
    )

    # Run session
    results = session.run_session(
        num_iterations=args.num_iterations,
        quick_feedback=not args.detailed_feedback,
        enable_playback=args.enable_playback,
        allow_regeneration=args.allow_regeneration,
    )

    # Export results
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/hil_session_{timestamp}")

    session.export_history(output_dir, export_csv=True)
    console.print(f"\n[green]✓ Session results saved to: {output_dir}[/green]")

    return 0


# ============================================================================
# Analyze Command
# ============================================================================


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze generated sequences."""
    from src.inference.sequence_analysis import SequenceAnalyzer, analyze_sequences

    console.print(
        Panel(
            "[bold cyan]Sequence Analysis[/bold cyan]",
            subtitle="Phase 3 Implementation",
        )
    )

    # Determine input source
    if args.sequences:
        sequences_path = Path(args.sequences)
    else:
        # Auto-detect from output directory
        output_dirs = list(Path("outputs").glob("inference_*/sequences.json"))
        if not output_dirs:
            console.print("[red]No sequences found. Specify --sequences path.[/red]")
            return 1
        sequences_path = sorted(output_dirs)[-1]
        console.print(f"[dim]Auto-detected: {sequences_path}[/dim]")

    # Create analyzer
    analyzer = SequenceAnalyzer(total_clusters=args.total_clusters)
    analyzer.load_sequences(sequences_path)

    # Generate report
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = sequences_path.parent / "analysis"

    analyzer.generate_report(
        output_dir,
        include_visualizations=not args.no_visualizations,
    )

    # Print summary
    metrics = analyzer.compute_all_metrics()
    diversity = metrics.get("diversity", {})

    table = Table(title="Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sequences", str(metrics.get("num_sequences", 0)))
    table.add_row("Unique Clusters", str(diversity.get("unique_clusters_total", 0)))
    table.add_row(
        "Avg Unique/Seq", f"{diversity.get('unique_per_sequence_mean', 0):.2f}"
    )
    table.add_row("Entropy", f"{diversity.get('entropy', 0):.3f}")
    table.add_row("Coverage", f"{diversity.get('coverage_ratio', 0):.1%}")

    console.print(table)
    console.print(f"\n[green]✓ Analysis saved to: {output_dir}[/green]")

    return 0


# ============================================================================
# Benchmark Command
# ============================================================================


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark comparison across checkpoints."""
    from src.inference.benchmark import BenchmarkRunner

    console.print(
        Panel(
            "[bold cyan]Benchmark Comparison[/bold cyan]",
            subtitle="Phase 4 Implementation",
        )
    )

    # Load base config
    config = load_config(args.config)

    # Create benchmark runner
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/benchmark_{timestamp}")

    runner = BenchmarkRunner(config, output_dir)

    # Add checkpoints
    if args.checkpoints_dir:
        # Auto-discover checkpoints
        checkpoints_dir = Path(args.checkpoints_dir)
        for cp_dir in sorted(checkpoints_dir.iterdir()):
            if cp_dir.is_dir():
                final_path = cp_dir / "checkpoints" / "final.pth"
                if not final_path.exists():
                    # Try nested structure
                    nested = list(cp_dir.glob("*/checkpoints/final.pth"))
                    if nested:
                        final_path = nested[0]

                if final_path.exists():
                    # Find config
                    config_candidates = [
                        cp_dir / "config.yaml",
                        cp_dir.parent / f"{cp_dir.name}.yaml",
                        list(cp_dir.glob("*.yaml")),
                    ]
                    config_path = None
                    for candidate in config_candidates:
                        if isinstance(candidate, list):
                            if candidate:
                                config_path = str(candidate[0])
                                break
                        elif Path(candidate).exists():
                            config_path = str(candidate)
                            break

                    runner.add_checkpoint(
                        name=cp_dir.name,
                        checkpoint_path=str(final_path),
                        config_path=config_path,
                    )

    if args.checkpoint:
        # Add single checkpoint
        runner.add_checkpoint(
            name=Path(args.checkpoint).parent.parent.name,
            checkpoint_path=args.checkpoint,
            config_path=args.checkpoint_config,
        )

    if not runner.checkpoints:
        console.print(
            "[red]No checkpoints found. Specify --checkpoints-dir or --checkpoint.[/red]"
        )
        return 1

    console.print(f"[cyan]Found {len(runner.checkpoints)} checkpoint(s)[/cyan]")

    # Run benchmarks
    results = runner.run_all(
        num_sequences=args.num_sequences,
        deterministic=not args.stochastic,
    )

    # Generate report
    runner.generate_comparison_report()

    # Print summary
    table = Table(title="Benchmark Results")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Reward", style="green")
    table.add_column("Diversity", style="yellow")
    table.add_column("Entropy", style="magenta")

    for r in results:
        table.add_row(
            r.checkpoint_name,
            f"{r.avg_episode_reward:.3f} ± {r.std_episode_reward:.3f}",
            f"{r.unique_per_sequence:.1f}",
            f"{r.entropy:.3f}",
        )

    console.print(table)
    console.print(f"\n[green]✓ Benchmark results saved to: {output_dir}[/green]")

    return 0


# ============================================================================
# Paper Results Command
# ============================================================================


def cmd_paper(args: argparse.Namespace) -> int:
    """Generate paper-ready results and figures."""
    from src.inference.paper_results import PaperResultsCompiler, generate_paper_results

    console.print(
        Panel(
            "[bold cyan]Paper Results Generation[/bold cyan]",
            subtitle="Phase 5 Implementation",
        )
    )

    # Determine input directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Try to find latest benchmark results
        benchmark_dirs = list(Path("outputs").glob("benchmark_*"))
        if not benchmark_dirs:
            console.print(
                "[red]No benchmark results found. Run benchmark first or specify --results-dir.[/red]"
            )
            return 1
        results_dir = sorted(benchmark_dirs)[-1]
        console.print(f"[dim]Using: {results_dir}[/dim]")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("nips_paper/figures/generated")

    # Compile results
    outputs = generate_paper_results(results_dir, output_dir)

    # Print summary
    console.print("\n[cyan]Generated Materials:[/cyan]")
    for material, path in outputs.items():
        if isinstance(path, list):
            for p in path:
                console.print(f"  • {p}")
        else:
            console.print(f"  • {path}")

    console.print(f"\n[green]✓ Paper materials saved to: {output_dir}[/green]")

    return 0


# ============================================================================
# Full Pipeline Command
# ============================================================================


# ============================================================================
# Simulation Commands
# ============================================================================


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run HIL preference-guided simulation."""
    from src.inference.simulation_runner import SimulationRunner
    from src.inference.cluster_profiles import load_default_profiles
    from src.inference.preference_simulation import get_predefined_scenarios
    from src.inference.config_validator import ConfigValidator

    # Load configuration
    config = load_config(args.config)

    # Validate configuration before running simulation
    checkpoint_path = Path(args.checkpoint)
    ghsom_dir = Path(
        config.get("ghsom", {}).get(
            "default_model_path",
            "experiments/ghsom_commu_full_tsne_optimized_20251125",
        )
    ).parent
    cluster_profiles_path = Path(
        config.get("cluster_profiles", {}).get(
            "default_path", "data/raw/cluster_profiles.csv"
        )
    )

    console.print("[dim]Validating configuration...[/dim]")
    validation_result = ConfigValidator.validate_checkpoint_config_consistency(
        checkpoint_path=checkpoint_path,
        config=config,
        ghsom_dir=ghsom_dir,
        cluster_profiles_path=cluster_profiles_path,
    )

    if not validation_result.valid:
        console.print("[red]Configuration validation failed:[/red]")
        for error in validation_result.errors:
            console.print(f"  [red]✗[/red] {error}")
        for warning in validation_result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")
        if validation_result.errors:
            return 1
    elif validation_result.warnings:
        console.print("[yellow]Configuration validation warnings:[/yellow]")
        for warning in validation_result.warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")

    console.print(
        Panel(
            "[bold cyan]HIL Preference-Guided Simulation[/bold cyan]",
            subtitle="De-learning Demonstration",
        )
    )

    # Load configuration
    config = load_config(args.config)

    # Load inference configuration if provided
    inference_config = None
    if args.inference_config:
        try:
            inference_config = InferenceConfig(args.inference_config)
            console.print(f"[dim]Using inference config: {args.inference_config}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to load inference config: {e}[/yellow]"
            )
            console.print("[dim]Using CLI arguments and defaults[/dim]")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    try:
        runner = SimulationRunner(
            checkpoint_path=Path(args.checkpoint),
            ghsom_dir=Path(
                config.get("ghsom", {}).get(
                    "default_model_path",
                    "experiments/ghsom_commu_full_tsne_optimized_20251125",
                )
            ).parent,
            output_dir=output_dir,
            config=config,
            inference_config=inference_config,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    # Determine scenarios to run
    available_scenarios = runner.get_available_scenarios()

    if args.scenario == "all":
        scenarios = available_scenarios
    else:
        if args.scenario not in available_scenarios:
            console.print(f"[red]Unknown scenario: {args.scenario}[/red]")
            console.print(f"[dim]Available: {available_scenarios}[/dim]")
            return 1
        scenarios = [args.scenario]

    # Use CLI args if provided, otherwise let config/defaults handle it
    num_iterations = args.iterations if hasattr(args, "iterations") else None
    num_seeds = args.seeds if hasattr(args, "seeds") else None
    adaptation_strength = (
        args.adaptation_strength if hasattr(args, "adaptation_strength") else None
    )

    console.print(f"[dim]Running scenarios: {scenarios}[/dim]")
    console.print(
        f"[dim]Iterations: {num_iterations or 'from config'}, "
        f"Seeds: {num_seeds or 'from config'}[/dim]"
    )
    console.print(
        f"[dim]Mode: {args.adaptation_mode}, "
        f"Strength: {adaptation_strength or 'from config'}[/dim]"
    )

    # Get policy learning and checkpoint saving options
    # CLI flags override config defaults
    enable_policy_learning = getattr(args, "enable_policy_learning", False)
    save_checkpoint = getattr(args, "save_checkpoint", False)

    # If not set via CLI, check inference config
    if not enable_policy_learning and inference_config is not None:
        enable_policy_learning = inference_config.get_policy_learning_enabled()
    if not save_checkpoint and inference_config is not None:
        save_checkpoint = inference_config.get_checkpoint_save_enabled()

    if enable_policy_learning:
        console.print("[cyan]Policy learning enabled[/cyan]")
    if save_checkpoint:
        console.print("[cyan]Checkpoint saving enabled[/cyan]")

    # Run simulations
    all_results = runner.run_all_scenarios(
        num_iterations=num_iterations,
        num_seeds=num_seeds,
        adaptation_mode=args.adaptation_mode,
        adaptation_strength=adaptation_strength,
        scenarios=scenarios,
        enable_policy_learning=enable_policy_learning,
        save_checkpoint=save_checkpoint,
        verbose=True,
    )

    # Run ablation if requested
    ablation_results = None
    if args.ablation and len(scenarios) > 0:
        console.print("\n[bold]Running ablation study...[/bold]")
        ablation_results = runner.run_ablation(
            scenario_name=scenarios[0],
            num_iterations=num_iterations,
            num_seeds=num_seeds,
            verbose=True,
        )

    # Run mode comparison if requested
    mode_results = None
    if args.mode_comparison and len(scenarios) > 0:
        console.print("\n[bold]Running mode comparison...[/bold]")
        mode_results = runner.run_mode_comparison(
            scenario_name=scenarios[0],
            num_iterations=num_iterations,
            num_seeds=num_seeds,
            adaptation_strength=adaptation_strength,
            verbose=True,
        )

    # Print summary
    runner.print_results_summary(all_results)

    # Generate visualizations
    console.print("\n[bold]Generating visualizations...[/bold]")
    from src.inference.simulation_visualizer import SimulationVisualizer

    figure_format = inference_config.get_figure_format() if inference_config else "png"
    visualizer = SimulationVisualizer(
        output_dir=output_dir / "figures",
        figure_format=figure_format,
    )

    # Convert results to dict format for visualizer
    results_dicts = {
        scenario: [r.to_dict() for r in results]
        for scenario, results in all_results.items()
    }

    ablation_dicts = None
    if ablation_results:
        ablation_dicts = {
            strength: [r.to_dict() for r in results]
            for strength, results in ablation_results.items()
        }

    mode_dicts = None
    if mode_results:
        mode_dicts = {
            mode: [r.to_dict() for r in results]
            for mode, results in mode_results.items()
        }

    figures = visualizer.generate_paper_figures(
        results_dicts,
        ablation_results=ablation_dicts,
        mode_results=mode_dicts,
    )

    visualizer.close_all()

    console.print(f"\n[green]✓ Simulation complete! Results in: {output_dir}[/green]")
    console.print(f"[dim]Generated {len(figures)} figures[/dim]")

    return 0


def cmd_visualize_simulation(args: argparse.Namespace) -> int:
    """Generate figures from simulation results."""
    import json
    from src.inference.simulation_visualizer import SimulationVisualizer

    console.print(
        Panel(
            "[bold cyan]Simulation Visualization[/bold cyan]",
            subtitle="Generate Paper Figures",
        )
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    if not results_dir.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return 1

    # Load results
    all_results = {}

    # Look for scenario results
    results_subdir = results_dir / "results"
    if results_subdir.exists():
        for scenario_dir in results_subdir.iterdir():
            if scenario_dir.is_dir():
                scenario_name = scenario_dir.name
                scenario_results = []

                for seed_dir in scenario_dir.iterdir():
                    if seed_dir.is_dir():
                        result_file = seed_dir / "result.json"
                        if result_file.exists():
                            with open(result_file, "r") as f:
                                scenario_results.append(json.load(f))

                if scenario_results:
                    all_results[scenario_name] = scenario_results

    if not all_results:
        console.print("[yellow]No simulation results found[/yellow]")
        return 1

    console.print(f"[dim]Found {len(all_results)} scenarios[/dim]")

    # Load ablation results if present
    ablation_results = None
    ablation_dir = results_dir / "ablation"
    if ablation_dir.exists():
        ablation_results = {}
        for strength_file in ablation_dir.glob("strength_*.json"):
            strength = float(strength_file.stem.split("_")[1])
            with open(strength_file, "r") as f:
                ablation_results[strength] = [json.load(f)]

    # Load mode comparison results if present
    mode_results = None
    mode_dir = results_dir / "mode_comparison"
    if mode_dir.exists():
        mode_results = {}
        for mode_file in mode_dir.glob("*_seed_*.json"):
            mode = mode_file.stem.rsplit("_seed_", 1)[0]
            if mode not in mode_results:
                mode_results[mode] = []
            with open(mode_file, "r") as f:
                mode_results[mode].append(json.load(f))

    # Generate visualizations
    visualizer = SimulationVisualizer(
        output_dir=output_dir,
        figure_format=args.format,
    )

    figures = visualizer.generate_paper_figures(
        all_results,
        ablation_results=ablation_results,
        mode_results=mode_results,
    )

    visualizer.close_all()

    console.print(
        f"\n[green]✓ Generated {len(figures)} figures in: {output_dir}[/green]"
    )

    return 0


def cmd_full(args: argparse.Namespace) -> int:
    """Run full inference pipeline: batch → analyze → benchmark → paper."""
    from scripts.inference.run_inference import main as run_inference_main

    console.print(
        Panel("[bold cyan]Full Inference Pipeline[/bold cyan]", subtitle="All Phases")
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = (
        Path(args.output) if args.output else Path(f"outputs/full_pipeline_{timestamp}")
    )
    base_output.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate sequences
    console.print("\n[bold]Step 1: Generating sequences...[/bold]")
    inference_output = base_output / "inference"

    # Use run_inference for batch generation
    sys.argv = [
        "run_inference.py",
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--mode",
        "batch",
        "--num-sequences",
        str(args.num_sequences),
        "--output",
        str(inference_output),
    ]
    run_inference_main()

    # Step 2: Analyze sequences
    console.print("\n[bold]Step 2: Analyzing sequences...[/bold]")
    args.sequences = str(inference_output / "sequences.json")
    args.output = str(base_output / "analysis")
    args.total_clusters = 22
    args.no_visualizations = False
    cmd_analyze(args)

    # Step 3: Generate paper results (if benchmark data exists)
    console.print("\n[bold]Step 3: Generating paper results...[/bold]")
    args.results_dir = str(base_output)
    args.output = str(base_output / "paper")
    try:
        cmd_paper(args)
    except Exception as e:
        console.print(f"[yellow]Paper generation skipped: {e}[/yellow]")

    console.print(
        f"\n[green]✓ Full pipeline complete! Results in: {base_output}[/green]"
    )

    return 0


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inference Pipeline with HIL Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # HIL command
    hil_parser = subparsers.add_parser("hil", help="Run interactive HIL session")
    hil_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    hil_parser.add_argument(
        "--config", default="configs/agent_config.yaml", help="Config file"
    )
    hil_parser.add_argument(
        "--num-iterations", type=int, default=10, help="Number of iterations"
    )
    hil_parser.add_argument(
        "--detailed-feedback", action="store_true", help="Use detailed feedback"
    )
    hil_parser.add_argument(
        "--enable-playback", action="store_true", help="Enable MIDI playback"
    )
    hil_parser.add_argument(
        "--allow-regeneration",
        action="store_true",
        default=True,
        help="Allow regeneration",
    )
    hil_parser.add_argument("--output", help="Output directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze sequences")
    analyze_parser.add_argument("--sequences", help="Path to sequences JSON")
    analyze_parser.add_argument("--output", help="Output directory")
    analyze_parser.add_argument(
        "--total-clusters", type=int, default=22, help="Total clusters"
    )
    analyze_parser.add_argument(
        "--no-visualizations", action="store_true", help="Skip visualizations"
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark comparison")
    bench_parser.add_argument("--checkpoints-dir", help="Directory with checkpoints")
    bench_parser.add_argument("--checkpoint", help="Single checkpoint path")
    bench_parser.add_argument(
        "--checkpoint-config", help="Config for single checkpoint"
    )
    bench_parser.add_argument(
        "--config", default="configs/agent_config.yaml", help="Base config"
    )
    bench_parser.add_argument(
        "--num-sequences", type=int, default=100, help="Sequences per checkpoint"
    )
    bench_parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic policy"
    )
    bench_parser.add_argument("--output", help="Output directory")

    # Paper command
    paper_parser = subparsers.add_parser("paper", help="Generate paper results")
    paper_parser.add_argument("--results-dir", help="Directory with benchmark results")
    paper_parser.add_argument("--output", help="Output directory")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    full_parser.add_argument(
        "--config", default="configs/agent_config.yaml", help="Config file"
    )
    full_parser.add_argument(
        "--num-sequences", type=int, default=100, help="Sequences to generate"
    )
    full_parser.add_argument("--output", help="Base output directory")

    # Simulate command (HIL preference-guided simulation)
    sim_parser = subparsers.add_parser(
        "simulate", help="Run HIL preference-guided simulation"
    )
    sim_parser.add_argument(
        "--checkpoint", required=True, help="Path to trained checkpoint"
    )
    sim_parser.add_argument(
        "--config", default="configs/agent_config.yaml", help="Config file"
    )
    sim_parser.add_argument(
        "--inference-config",
        help="Inference config file (default: configs/inference_config.yaml)",
    )
    sim_parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario name or 'all' for all scenarios",
    )
    sim_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iterations per simulation (default: from config)",
    )
    sim_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of random seeds (default: from config)",
    )
    sim_parser.add_argument(
        "--adaptation-mode",
        choices=["q_penalty", "reward_shaping"],
        default=None,
        help="Adaptation mode (default: from config)",
    )
    sim_parser.add_argument(
        "--adaptation-strength",
        type=float,
        default=None,
        help="Adaptation strength (default: from config)",
    )
    sim_parser.add_argument(
        "--ablation", action="store_true", help="Run ablation study"
    )
    sim_parser.add_argument(
        "--mode-comparison", action="store_true", help="Compare adaptation modes"
    )
    sim_parser.add_argument(
        "--enable-policy-learning",
        action="store_true",
        help="Enable gradient-based policy updates during simulation",
    )
    sim_parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save adapted checkpoint after simulation",
    )
    sim_parser.add_argument(
        "--output", default="outputs/hil_simulation", help="Output directory"
    )

    # Visualize simulation command
    vis_parser = subparsers.add_parser(
        "visualize-simulation", help="Generate figures from simulation results"
    )
    vis_parser.add_argument(
        "--results-dir", required=True, help="Simulation results directory"
    )
    vis_parser.add_argument(
        "--output", default="outputs/hil_figures", help="Figure output directory"
    )
    vis_parser.add_argument(
        "--format", choices=["pdf", "png"], default="pdf", help="Figure format"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    handlers = {
        "hil": cmd_hil,
        "analyze": cmd_analyze,
        "benchmark": cmd_benchmark,
        "paper": cmd_paper,
        "full": cmd_full,
        "simulate": cmd_simulate,
        "visualize-simulation": cmd_visualize_simulation,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        logger.error(f"Pipeline error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
