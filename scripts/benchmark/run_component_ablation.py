#!/usr/bin/env python
"""Component Ablation Runner.

Runs ablation experiments for various system components.

Ablation Groups:
1. Coordination Ablation (Table 3): observation embeddings, topology rewards
2. Curriculum Ablation (Table 4): curriculum stages
3. Reward Ablation (Table 7): reward components
4. Diversity Grid Search (Table 8): diversity range optimization
5. HIL Layer Ablation (Table 9): three-layer adaptation mechanism

Usage:
    # Run all training ablations (coordination, curriculum, reward, diversity)
    python scripts/benchmark/run_component_ablation.py --all-training

    # Run specific ablation group
    python scripts/benchmark/run_component_ablation.py --quick --ablation coordination -j 4
    python scripts/benchmark/run_component_ablation.py --quick --ablation curriculum -j 4
    python scripts/benchmark/run_component_ablation.py --quick --ablation reward -j 4
    python scripts/benchmark/run_component_ablation.py --quick --ablation diversity -j 4

    # Run HIL layer ablation (auto-discovers best checkpoint)
    python scripts/benchmark/run_component_ablation.py --quick --ablation hil -j 4

    # Run HIL with specific checkpoint
    python scripts/benchmark/run_component_ablation.py --ablation hil --checkpoint <path>

    # Run HIL with different model type
    python scripts/benchmark/run_component_ablation.py --ablation hil --model-type baseline_cl

    # List available checkpoints
    python scripts/benchmark/run_component_ablation.py --list-checkpoints

    # Quick run (3 seeds) vs Full run (5 seeds)
    python scripts/benchmark/run_component_ablation.py --ablation coordination --quick
    python scripts/benchmark/run_component_ablation.py --ablation coordination --full

    # Dry run to see commands
    python scripts/benchmark/run_component_ablation.py --ablation coordination --dry-run

    # Parallel execution
    python scripts/benchmark/run_component_ablation.py --ablation coordination -j 4

Target Tables:
    - Table 3 (coordination_ablation): Coordination mechanisms ablation
    - Table 4 (curriculum_ablation): Curriculum learning stages
    - Table 7 (reward_ablation): Reward function components
    - Table 8 (diversity_grid): Diversity range grid search
    - Table 9 (layer_ablation): HIL three-layer adaptation

Checkpoint Auto-Discovery:
    For HIL ablation, the script auto-discovers checkpoints from previous benchmark runs.
    Default model: Dueling+PER with Curriculum Learning (best performing model).
    Supported model types: dueling_per_cl, dueling_cl, baseline_cl, baseline
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

# Base path for ablation configs
CONFIG_BASE = "configs/ablations"

# Ablation groups with their configs
ABLATION_CONFIGS = {
    # Table 3: Coordination Mechanisms Ablation
    # Note: Only prototype and centroid modes are implemented (see src/agents/cluster_feature_mapper.py:86-91)
    # "one_hot" mode is NOT implemented - removed from ablation configs
    # "raw_centroid_embeddings" REMOVED: requires GHSOM trained on raw features (incompatible dimensions)
    "coordination": {
        "prototype_embeddings": f"{CONFIG_BASE}/coordination/prototype_embeddings.yaml",
        "centroid_embeddings": f"{CONFIG_BASE}/coordination/centroid_embeddings.yaml",
        "no_topology_rewards": f"{CONFIG_BASE}/coordination/no_topology_rewards.yaml",
    },
    # Table 4: Curriculum Learning Ablation
    "curriculum": {
        "no_curriculum_flat": f"{CONFIG_BASE}/curriculum/no_curriculum_flat.yaml",
        "two_stage_curriculum": f"{CONFIG_BASE}/curriculum/two_stage_curriculum.yaml",
        "three_stage_curriculum": f"{CONFIG_BASE}/curriculum/three_stage_curriculum.yaml",
    },
    # Table 7: Reward Components Ablation
    "reward": {
        "full_reward": f"{CONFIG_BASE}/reward/full_reward.yaml",
        "no_structure": f"{CONFIG_BASE}/reward/no_structure.yaml",
        "no_transition": f"{CONFIG_BASE}/reward/no_transition.yaml",
        "no_diversity": f"{CONFIG_BASE}/reward/no_diversity.yaml",
        "terminal_only": f"{CONFIG_BASE}/reward/terminal_only.yaml",
    },
    # Table 8: Diversity Range Grid Search
    "diversity": {
        "range_050_062": f"{CONFIG_BASE}/diversity/range_0.50_0.62.yaml",
        "range_056_069": f"{CONFIG_BASE}/diversity/range_0.56_0.69.yaml",
        "range_062_075": f"{CONFIG_BASE}/diversity/range_0.62_0.75.yaml",
        "range_069_081": f"{CONFIG_BASE}/diversity/range_0.69_0.81.yaml",
        "range_075_088": f"{CONFIG_BASE}/diversity/range_0.75_0.88.yaml",
    },
    # Table 9: HIL Three-Layer Ablation (Inference)
    "hil": {
        "all_three_layers": f"{CONFIG_BASE}/hil/all_three_layers.yaml",
        "layer3_only": f"{CONFIG_BASE}/hil/layer3_only.yaml",
        "layers12_only": f"{CONFIG_BASE}/hil/layers12_only.yaml",
        "layer1_only": f"{CONFIG_BASE}/hil/layer1_only.yaml",
        "layer2_only": f"{CONFIG_BASE}/hil/layer2_only.yaml",
        "no_adaptation": f"{CONFIG_BASE}/hil/no_adaptation.yaml",
    },
}

# Table mapping for documentation
TABLE_MAPPING = {
    "coordination": "Table 3 (coordination_ablation)",
    "curriculum": "Table 4 (curriculum_ablation)",
    "reward": "Table 7 (reward_ablation)",
    "diversity": "Table 8 (diversity_grid)",
    "hil": "Table 9 (layer_ablation)",
}

# Seed configurations
QUICK_SEEDS = [42, 123, 456]
FULL_SEEDS = [42, 123, 456, 789, 202]

# HIL scenarios for layer ablation
HIL_SCENARIOS = [
    "calm_relaxation",
    "energetic_drive",
    "piano_focus",
    "strings_ensemble",
    "melodic_focus",
    "ambient_background",
    "intrinsic_feel",
]

# =============================================================================
# CHECKPOINT CONFIGURATION FOR HIL ABLATION
# =============================================================================

# Default checkpoint search patterns (in priority order)
# The best model is Dueling+PER with Curriculum Learning
DEFAULT_CHECKPOINT_PATTERNS = [
    # NIPS benchmark outputs - best model (Dueling+PER CL)
    "artifacts/nips_benchmark/*/dueling_per_cl_*/checkpoints/*.pth",
    "outputs/nips_benchmark/*/dueling_per_cl_*/policy.pth",
    "outputs/benchmark_reports/nips_benchmark/*/checkpoints/dueling_per_cl*.pth",
    # Ablation benchmark outputs
    "outputs/ablation_benchmark/*/checkpoints/*.pth",
    # General training outputs
    "artifacts/training/*/checkpoints/*.pth",
    "artifacts/*/checkpoints/*.pth",
]

# Known checkpoint locations (manually verified best models)
# Paths can be absolute or relative (relative paths use glob patterns)
KNOWN_CHECKPOINTS = {
    "dueling_per_cl": {
        "description": "Dueling+PER with Curriculum Learning (Best Model)",
        "paths": [
            # Absolute path to verified best model
            "artifacts/benchmark/nips_benchmark/20251216_140256/runs/dueling_per_cl_s42/run_dueling_drqn_20251216_214013/checkpoints/final.pth",
            # Glob patterns for discovery
            "artifacts/benchmark/nips_benchmark/*/runs/dueling_per_cl_*/*/checkpoints/final.pth",
            "artifacts/nips_benchmark/*/dueling_per_cl_*/checkpoints/*.pth",
        ],
    },
    "baseline_cl": {
        "description": "Baseline DRQN with Curriculum Learning",
        "paths": [
            "artifacts/benchmark/nips_benchmark/*/runs/baseline_cl_*/*/checkpoints/final.pth",
            "outputs/nips_benchmark/*/baseline_cl_*/checkpoints/best_policy.pth",
        ],
    },
    "dueling_cl": {
        "description": "Dueling DQN with Curriculum Learning",
        "paths": [
            "artifacts/benchmark/nips_benchmark/*/runs/dueling_cl_*/*/checkpoints/final.pth",
            "outputs/nips_benchmark/*/dueling_cl_*/checkpoints/best_policy.pth",
        ],
    },
}


def find_checkpoints(base_dir: Path = None) -> Dict[str, List[Path]]:
    """Auto-discover available checkpoints from benchmark outputs.

    Args:
        base_dir: Base directory to search (defaults to project root)

    Returns:
        Dictionary mapping model name to list of checkpoint paths
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    discovered = {}

    # Search known checkpoint locations first
    for model_name, config in KNOWN_CHECKPOINTS.items():
        paths = []
        for pattern in config["paths"]:
            # Check if it's an absolute path or contains glob wildcards
            if "*" in pattern:
                # Glob pattern - use glob
                matches = list(base_dir.glob(pattern))
                paths.extend([p for p in matches if p.exists()])
            else:
                # Direct path - check if it exists
                direct_path = base_dir / pattern
                if direct_path.exists():
                    paths.append(direct_path)
        if paths:
            discovered[model_name] = sorted(
                paths, key=lambda p: p.stat().st_mtime, reverse=True
            )

    # Search default patterns
    for pattern in DEFAULT_CHECKPOINT_PATTERNS:
        matches = list(base_dir.glob(pattern))
        for match in matches:
            # Extract model name from path
            parts = match.parts
            for i, part in enumerate(parts):
                if "dueling_per_cl" in part:
                    key = "dueling_per_cl"
                    break
                elif "dueling_cl" in part and "per" not in part:
                    key = "dueling_cl"
                    break
                elif "baseline_cl" in part:
                    key = "baseline_cl"
                    break
                elif "baseline" in part and "cl" not in part:
                    key = "baseline"
                    break
            else:
                key = "unknown"

            if key not in discovered:
                discovered[key] = []
            if match not in discovered[key]:
                discovered[key].append(match)

    # Sort by modification time (newest first)
    for key in discovered:
        discovered[key] = sorted(
            discovered[key],
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )

    return discovered


def get_default_checkpoint(model_type: str = "dueling_per_cl") -> Optional[Path]:
    """Get the default checkpoint path for HIL ablation.

    Args:
        model_type: Type of model checkpoint to find (default: dueling_per_cl)

    Returns:
        Path to checkpoint or None if not found
    """
    discovered = find_checkpoints()

    # Prefer the specified model type
    if model_type in discovered and discovered[model_type]:
        return discovered[model_type][0]

    # Fallback to best model (dueling_per_cl)
    if "dueling_per_cl" in discovered and discovered["dueling_per_cl"]:
        return discovered["dueling_per_cl"][0]

    # Fallback to any available checkpoint
    for model, paths in discovered.items():
        if paths:
            print(f"⚠️ Default checkpoint not found, using {model}: {paths[0]}")
            return paths[0]

    return None


def list_available_checkpoints() -> None:
    """Print all available checkpoints."""
    discovered = find_checkpoints()

    print("\n" + "=" * 60)
    print("AVAILABLE CHECKPOINTS FOR HIL ABLATION")
    print("=" * 60)

    if not discovered:
        print("❌ No checkpoints found!")
        print(
            "   Run training first: python scripts/nips_benchmark/run_nips_benchmark.py"
        )
        return

    for model_type, paths in sorted(discovered.items()):
        desc = KNOWN_CHECKPOINTS.get(model_type, {}).get("description", model_type)
        print(f"\n📁 {model_type}: {desc}")
        for i, path in enumerate(paths[:3]):  # Show max 3 per type
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
            marker = "★" if i == 0 else " "
            print(f"   {marker} {path} [{mtime}]")
        if len(paths) > 3:
            print(f"   ... and {len(paths) - 3} more")

    print("\n" + "=" * 60)


def run_training_experiment(
    variant: str,
    seed: int,
    output_dir: Path,
    config_path: str,
    dry_run: bool = False,
    no_wandb: bool = False,
) -> dict:
    """Run a single training experiment.

    Args:
        variant: Variant name
        seed: Random seed
        output_dir: Base output directory
        config_path: Path to configuration file
        dry_run: If True, print command without executing
        no_wandb: If True, disable WandB logging

    Returns:
        dict with experiment result
    """
    exp_dir = output_dir / "runs" / f"{variant}_s{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/training/run_training.py",
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--output-dir",
        str(exp_dir),
        "--wandb-tags",
        f"ablation_benchmark,{variant},seed{seed}",
    ]

    if no_wandb:
        cmd.append("--no-wandb")

    result = {
        "variant": variant,
        "seed": seed,
        "config": config_path,
        "output_dir": str(exp_dir),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "final_reward": None,
        "best_reward": None,
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"ABLATION BENCHMARK: {variant.upper()} (seed={seed})")
    print(f"Config: {config_path}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}\n")

    result["start_time"] = datetime.now().isoformat()

    try:
        process = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "completed"

            # Load metrics from saved file
            # Find the nested run_*/ subdirectory
            run_dirs = list(exp_dir.glob("run_*/"))
            if run_dirs:
                run_dir = run_dirs[0]  # Use the first (should only be one)
                metrics_file = run_dir / "metrics" / "comprehensive" / "training_summary.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    # Keys are in format "summary/best_episode_reward"
                    result["best_reward"] = metrics.get("summary/best_episode_reward")
                    result["final_reward"] = metrics.get("summary/final_mean_reward")

            print(
                f"✅ {variant} (seed={seed}) completed - Best: {result.get('best_reward', 'N/A')}"
            )
        else:
            result["status"] = "failed"
            result["error"] = (
                process.stderr[-500:] if process.stderr else "Unknown error"
            )
            print(f"❌ {variant} (seed={seed}) failed: {result['error'][:200]}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment exceeded 2 hour timeout"
        result["end_time"] = datetime.now().isoformat()
        print(f"⏰ {variant} (seed={seed}) timed out")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"❌ {variant} (seed={seed}) error: {e}")

    # Save result
    result_file = exp_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_hil_experiment(
    variant: str,
    scenario: str,
    seed: int,
    output_dir: Path,
    config_path: str,
    checkpoint_path: str,
    dry_run: bool = False,
) -> dict:
    """Run a single HIL inference experiment.

    Args:
        variant: Variant name (layer configuration)
        scenario: Preference scenario name
        seed: Random seed
        output_dir: Base output directory
        config_path: Path to HIL configuration file
        checkpoint_path: Path to trained model checkpoint
        dry_run: If True, print command without executing

    Returns:
        dict with experiment result
    """
    exp_dir = output_dir / "runs" / f"{variant}_{scenario}_s{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Use simulate subcommand with proper arguments
    # Note: simulate uses --seeds as count (not specific value), so we run 1 seed at a time
    # and let the outer loop handle multiple iterations for averaging
    # HIL configs are inference configs, so use --inference-config not --config
    cmd = [
        sys.executable,
        "scripts/inference/run_inference_pipeline.py",
        "simulate",
        "--checkpoint",
        checkpoint_path,
        "--scenario",
        scenario,
        "--inference-config",
        config_path,
        "--seeds",
        "1",  # Run 1 seed per invocation
        "--output",
        str(exp_dir),
    ]

    result = {
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "output_dir": str(exp_dir),
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "feedback_improvement": None,
        "desirable_delta": None,
        "learning_detected": None,
        "error": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"\n{'='*60}")
    print(f"HIL ABLATION: {variant.upper()} - {scenario} (seed={seed})")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    result["start_time"] = datetime.now().isoformat()

    try:
        process = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for HIL
        )

        result["end_time"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "completed"

            # Load HIL results from simulate output structure
            # simulate saves to: output_dir/results/{scenario}/seed_0/result.json
            hil_result_file = exp_dir / "results" / scenario / "seed_0" / "result.json"
            if hil_result_file.exists():
                with open(hil_result_file) as f:
                    hil_data = json.load(f)
                result["feedback_improvement"] = hil_data.get("feedback_improvement")
                # Compute desirable_delta from initial and final ratios
                initial_des = hil_data.get("initial_desirable_ratio", 0)
                final_des = hil_data.get("final_desirable_ratio", 0)
                result["desirable_delta"] = final_des - initial_des
                # Distribution shift can indicate learning
                result["learning_detected"] = hil_data.get("distribution_shift", 0) > 0.1
            else:
                # Try alternative path in case of structure change
                alt_result_file = exp_dir / "simulation_result.json"
                if alt_result_file.exists():
                    with open(alt_result_file) as f:
                        hil_data = json.load(f)
                    result["feedback_improvement"] = hil_data.get("feedback_improvement")
                    initial_des = hil_data.get("initial_desirable_ratio", 0)
                    final_des = hil_data.get("final_desirable_ratio", 0)
                    result["desirable_delta"] = final_des - initial_des
                    result["learning_detected"] = hil_data.get("distribution_shift", 0) > 0.1

            print(f"✅ {variant}/{scenario} (seed={seed}) completed")
        else:
            result["status"] = "failed"
            result["error"] = (
                process.stderr[-500:] if process.stderr else "Unknown error"
            )
            print(f"❌ {variant}/{scenario} (seed={seed}) failed")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Experiment exceeded 30 minute timeout"
        result["end_time"] = datetime.now().isoformat()
        print(f"⏰ {variant}/{scenario} (seed={seed}) timed out")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        print(f"❌ {variant}/{scenario} (seed={seed}) error: {e}")

    # Save result
    result_file = exp_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_experiment_wrapper(args: Tuple) -> dict:
    """Wrapper for parallel execution."""
    exp_type = args[0]
    if exp_type == "training":
        _, variant, seed, output_dir, config_path, dry_run, no_wandb = args
        return run_training_experiment(
            variant, seed, output_dir, config_path, dry_run, no_wandb
        )
    elif exp_type == "hil":
        (
            _,
            variant,
            scenario,
            seed,
            output_dir,
            config_path,
            checkpoint_path,
            dry_run,
        ) = args
        return run_hil_experiment(
            variant, scenario, seed, output_dir, config_path, checkpoint_path, dry_run
        )


def run_parallel(experiments: List[Tuple], workers: int) -> List[dict]:
    """Run experiments in parallel."""
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment_wrapper, exp): exp for exp in experiments
        }

        for future in as_completed(future_to_exp):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                exp = future_to_exp[future]
                results.append(
                    {
                        "variant": exp[1] if len(exp) > 1 else "unknown",
                        "status": "error",
                        "error": str(e),
                    }
                )
                print(f"❌ Parallel execution error: {e}")

    return results


def generate_latex_table(ablation_type: str, results: List[dict], output_dir: Path):
    """Generate LaTeX table from results."""
    import statistics

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Group results by variant
    variants = {}
    for r in results:
        if r["status"] != "completed":
            continue
        variant = r["variant"]
        if variant not in variants:
            variants[variant] = []
        if "best_reward" in r and r["best_reward"] is not None:
            variants[variant].append(r["best_reward"])
        elif "feedback_improvement" in r and r["feedback_improvement"] is not None:
            variants[variant].append(r["feedback_improvement"])

    # Generate table content
    table_lines = []
    for variant, rewards in sorted(variants.items()):
        if rewards:
            mean = statistics.mean(rewards)
            std = statistics.stdev(rewards) if len(rewards) > 1 else 0
            table_lines.append(
                f"{variant.replace('_', ' ').title()} & {mean:.3f} & {std:.3f} \\\\"
            )

    # Write to file
    table_file = tables_dir / f"{ablation_type}_ablation.tex"
    with open(table_file, "w") as f:
        f.write(
            f"% Auto-generated LaTeX table for {TABLE_MAPPING.get(ablation_type, ablation_type)}\n"
        )
        f.write(f"% Generated: {datetime.now().isoformat()}\n\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Variant} & \\textbf{Reward} & \\textbf{Std} \\\\\n")
        f.write("\\midrule\n")
        for line in table_lines:
            f.write(line + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"📄 LaTeX table saved: {table_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Paper Ablation Benchmark for experiments section tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target Tables:
    - Table 3 (coordination): Coordination mechanisms ablation
    - Table 4 (curriculum): Curriculum learning stages
    - Table 7 (reward): Reward function components
    - Table 8 (diversity): Diversity range grid search
    - Table 9 (hil): HIL three-layer adaptation

Examples:
    # Run coordination ablation (Table 3)
    python scripts/ablation_benchmark/run_paper_ablation_benchmark.py --ablation coordination --quick

    # Run all training ablations
    python scripts/ablation_benchmark/run_paper_ablation_benchmark.py --all-training --quick

    # Run HIL layer ablation (requires checkpoint)
    python scripts/ablation_benchmark/run_paper_ablation_benchmark.py --ablation hil --checkpoint <path>

    # Dry run
    python scripts/ablation_benchmark/run_paper_ablation_benchmark.py --ablation reward --dry-run
        """,
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation group to run",
    )
    parser.add_argument(
        "--all-training",
        action="store_true",
        help="Run all training ablations (coordination, curriculum, reward, diversity)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with 3 seeds",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run with 5 seeds",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Custom random seeds",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Specific variants to run within an ablation group",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (auto-discovers if not specified)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="dueling_per_cl",
        choices=["dueling_per_cl", "dueling_cl", "baseline_cl", "baseline"],
        help="Model type for HIL ablation (default: dueling_per_cl - best model)",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="HIL scenarios to test (default: all 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Run N experiments in parallel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip generating LaTeX tables",
    )
    args = parser.parse_args()

    # Handle --list-checkpoints
    if args.list_checkpoints:
        list_available_checkpoints()
        sys.exit(0)

    # Validate arguments
    if not args.ablation and not args.all_training:
        parser.error("Either --ablation or --all-training is required")

    # Handle HIL checkpoint - auto-discover if not specified
    if args.ablation == "hil":
        if not args.checkpoint:
            # Try to auto-discover checkpoint
            default_ckpt = get_default_checkpoint(args.model_type)
            if default_ckpt:
                args.checkpoint = str(default_ckpt)
                print(f"✅ Auto-discovered checkpoint: {args.checkpoint}")
                print(f"   Model type: {args.model_type}")
            else:
                print("❌ No checkpoint found for HIL ablation!")
                print("   Options:")
                print("   1. Specify checkpoint: --checkpoint <path>")
                print(
                    "   2. Run training first: python scripts/nips_benchmark/run_nips_benchmark.py"
                )
                print("   3. List available: --list-checkpoints")
                sys.exit(1)
        else:
            # Verify provided checkpoint exists
            if not Path(args.checkpoint).exists():
                parser.error(f"Checkpoint not found: {args.checkpoint}")

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.full:
        seeds = FULL_SEEDS
    elif args.quick:
        seeds = QUICK_SEEDS
    else:
        seeds = [42]  # Default: single seed

    # Determine ablation groups
    if args.all_training:
        ablation_groups = ["coordination", "curriculum", "reward", "diversity"]
    else:
        ablation_groups = [args.ablation]

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/ablation_benchmark/{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify configs exist
    base_dir = Path(__file__).parent.parent.parent
    for group in ablation_groups:
        configs = ABLATION_CONFIGS[group]
        variants_to_check = args.variants if args.variants else configs.keys()
        for variant in variants_to_check:
            if variant not in configs:
                print(f"⚠️ Unknown variant: {variant}")
                continue
            config_path = base_dir / configs[variant]
            if not config_path.exists():
                print(f"❌ Config not found: {config_path}")
                sys.exit(1)

    # Print summary
    total_experiments = 0
    print(f"\n{'='*60}")
    print("PAPER ABLATION BENCHMARK")
    print(f"{'='*60}")
    print(f"Ablation Groups: {ablation_groups}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")
    print(f"Parallel: {args.parallel if args.parallel else 'Sequential'}")

    for group in ablation_groups:
        configs = ABLATION_CONFIGS[group]
        variants = args.variants if args.variants else list(configs.keys())
        n_variants = len([v for v in variants if v in configs])
        if group == "hil":
            scenarios = args.scenarios if args.scenarios else HIL_SCENARIOS
            n_exp = n_variants * len(scenarios) * len(seeds)
        else:
            n_exp = n_variants * len(seeds)
        total_experiments += n_exp
        print(
            f"  {group}: {n_variants} variants × {len(seeds)} seeds = {n_exp} experiments"
        )
        print(f"    Target: {TABLE_MAPPING.get(group, group)}")

    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}\n")

    all_results = {}

    # Run experiments for each ablation group
    for group in ablation_groups:
        configs = ABLATION_CONFIGS[group]
        variants = args.variants if args.variants else list(configs.keys())

        print(f"\n{'='*60}")
        print(f"RUNNING: {group.upper()} ABLATION")
        print(f"Target: {TABLE_MAPPING.get(group, group)}")
        print(f"{'='*60}")

        experiments = []

        if group == "hil":
            # HIL experiments need checkpoint and scenarios
            scenarios = args.scenarios if args.scenarios else HIL_SCENARIOS
            for variant in variants:
                if variant not in configs:
                    continue
                config_path = configs[variant]
                for scenario in scenarios:
                    for seed in seeds:
                        experiments.append(
                            (
                                "hil",
                                variant,
                                scenario,
                                seed,
                                output_dir / group,
                                config_path,
                                args.checkpoint,
                                args.dry_run,
                            )
                        )
        else:
            # Training experiments
            for variant in variants:
                if variant not in configs:
                    continue
                config_path = configs[variant]
                for seed in seeds:
                    experiments.append(
                        (
                            "training",
                            variant,
                            seed,
                            output_dir / group,
                            config_path,
                            args.dry_run,
                            args.no_wandb,
                        )
                    )

        # Run experiments
        if args.parallel and not args.dry_run:
            print(
                f"🚀 Running {len(experiments)} experiments in parallel ({args.parallel} workers)..."
            )
            results = run_parallel(experiments, args.parallel)
        else:
            results = []
            for exp in experiments:
                result = run_experiment_wrapper(exp)
                results.append(result)

        all_results[group] = results

        # Generate LaTeX table
        if not args.skip_tables and not args.dry_run:
            generate_latex_table(group, results, output_dir / group)

    # Save comprehensive summary
    summary_file = output_dir / "ablation_benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "benchmark_type": "paper_ablation",
                "ablation_groups": ablation_groups,
                "seeds": seeds,
                "timestamp": datetime.now().isoformat(),
                "results": {k: v for k, v in all_results.items()},
            },
            f,
            indent=2,
        )

    # Print final summary
    print(f"\n{'='*60}")
    print("ABLATION BENCHMARK COMPLETE")
    print(f"{'='*60}")

    for group, results in all_results.items():
        completed = len([r for r in results if r.get("status") == "completed"])
        failed = len(
            [r for r in results if r.get("status") in ("failed", "error", "timeout")]
        )
        print(f"{group}: {completed} completed, {failed} failed")

    print(f"\nSummary saved: {summary_file}")
    print(f"Tables saved: {output_dir}/*/tables/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
