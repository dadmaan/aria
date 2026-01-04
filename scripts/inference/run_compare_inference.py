#!/usr/bin/env python3
"""compare_algorithms.py - Compare different trained models."""
import json
from pathlib import Path
from scripts.inference.run_inference import (
    load_config,
    setup_environment,
    load_agent,
    run_batch_generation,
    calculate_aggregate_metrics,
)

CHECKPOINTS = {
    "DRQN-100": "artifacts/training/run_drqn_20251207_195352/checkpoints/final.pth",
    "DRQN-500K": "artifacts/training/extended_training/e2_500k_s42_251207_0011/run_drqn_20251207_001128/checkpoints/final.pth",
    "C51": "artifacts/training/run_c51_drqn_20251207_195249/checkpoints/final.pth",
    "Rainbow": "artifacts/training/run_rainbow_drqn_20251207_195319/checkpoints/final.pth",
}

results = {}

for name, checkpoint_path in CHECKPOINTS.items():
    print(f"\nEvaluating {name}...")

    # Load config from checkpoint
    config = load_config("configs/agent_config.yaml")
    env, ghsom = setup_environment(config)
    trainer = load_agent(checkpoint_path, env, config)

    # Generate sequences
    gen_results = run_batch_generation(trainer, env, num_sequences=20, verbose=0)
    metrics = calculate_aggregate_metrics(gen_results)

    results[name] = {
        "avg_reward": metrics["avg_reward"],
        "unique_clusters": metrics["unique_clusters_total"],
        "unique_per_seq": metrics["unique_per_sequence_mean"],
    }

    print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"  Unique Clusters: {metrics['unique_clusters_total']}")

# Save comparison
with open("outputs/algorithm_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nComparison saved to outputs/algorithm_comparison.json")
