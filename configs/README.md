# Configuration Files

This directory contains configuration files for training, inference, and benchmark experiments.

## Directory Structure

```
configs/
├── benchmark/
│   └── main/                    # Main 2×2 factorial benchmark configs
│       ├── baseline_no_cl.yaml
│       ├── baseline_cl.yaml
│       ├── rainbow_no_cl.yaml
│       ├── rainbow_cl.yaml
│       └── ...                  # Additional algorithm variants
├── ablations/
│   ├── algorithm/               # Algorithm ablation configs (DQN variants)
│   │   ├── baseline.yaml
│   │   ├── dueling_drqn.yaml
│   │   ├── c51.yaml
│   │   └── rainbow_drqn.yaml
│   ├── coordination/            # GHSOM coordination ablations
│   ├── curriculum/              # Curriculum learning ablations
│   ├── reward/                  # Reward component ablations
│   ├── diversity/               # Diversity range grid search
│   └── hil/                     # HIL layer ablations
├── development/                 # Development/experimental configs
├── training.yaml                # Default training configuration
├── inference.yaml               # Default HIL inference configuration
└── README.md
```

## Main Configuration Files

### `training.yaml`
Default production training configuration combining learnings from benchmark experiments.

**Key Parameters:**
- Algorithm: DQN with Double DQN enabled
- Network: DRQN (Recurrent DQN with LSTM h=256)
- Training: 200,000 timesteps, learning_rate=0.0001, gamma=0.95
- Reward: Structure(0.25) + Transition(0.5) + Diversity(0.25)

### `inference.yaml`
Human-in-the-loop (HIL) inference and simulation configuration.

**Key Parameters:**
- Feedback simulation with realistic noise models
- Three-layer adaptation: Q-penalty, reward shaping, policy learning
- Pre-defined preference scenarios

## Benchmark Configurations

### Main Benchmark (`benchmark/main/`)
Configs for the 2×2 factorial design (Algorithm × Curriculum Learning):

| Config | Description |
|--------|-------------|
| `baseline_no_cl.yaml` | Base DQN without curriculum |
| `baseline_cl.yaml` | Base DQN with curriculum |
| `rainbow_no_cl.yaml` | Rainbow DQN without curriculum |
| `rainbow_cl.yaml` | Rainbow DQN with curriculum |

Additional variants include Dueling, PER, NoisyNet, and combinations.

**Usage:**
```bash
python scripts/benchmark/run_main_benchmark.py --quick
```

## Ablation Configurations

### Algorithm Ablations (`ablations/algorithm/`)
Compare DQN algorithm variants:
- `baseline.yaml` - Standard DQN
- `dueling_drqn.yaml` - Dueling architecture
- `c51.yaml` - Categorical distributional RL
- `rainbow_drqn.yaml` - Full Rainbow

### Reward Ablations (`ablations/reward/`)
Test individual reward component contributions:
- `full_reward.yaml` - All components enabled
- `no_structure.yaml` - Structure component disabled
- `no_transition.yaml` - Transition component disabled
- `no_diversity.yaml` - Diversity component disabled
- `terminal_only.yaml` - Only terminal reward

### Curriculum Ablations (`ablations/curriculum/`)
Compare curriculum learning strategies:
- `no_curriculum_flat.yaml` - Full action space from start
- `two_stage_curriculum.yaml` - 2-phase progressive learning
- `three_stage_curriculum.yaml` - 3-phase progressive learning

### Coordination Ablations (`ablations/coordination/`)
Test GHSOM integration strategies:
- `prototype_embeddings.yaml` - Use prototype vectors
- `centroid_embeddings.yaml` - Use cluster centroids
- `no_topology_rewards.yaml` - Disable GHSOM-based rewards

### HIL Layer Ablations (`ablations/hil/`)
Test three-layer adaptation mechanisms:
- `all_three_layers.yaml` - Full adaptation
- `layer1_only.yaml` - Q-penalty only
- `layer2_only.yaml` - Reward shaping only
- `layer3_only.yaml` - Policy learning only

## Config-Script Mapping

| Script | Config Location |
|--------|----------------|
| `run_main_benchmark.py` | `configs/benchmark/main/` |
| `run_algorithm_ablation.py` | `configs/ablations/algorithm/` |
| `run_curriculum_ablation.py` | `configs/ablations/curriculum/` |
| `run_component_ablation.py` | `configs/ablations/*` |

## Creating Custom Configurations

1. Start from an existing config:
   ```bash
   cp configs/training.yaml configs/my_experiment.yaml
   ```

2. Modify parameters as needed

3. Run with custom config:
   ```bash
   python scripts/training/run_training.py --config configs/my_experiment.yaml
   ```

## Key Configuration Parameters

### Training Parameters
| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `total_timesteps` | Training budget | 50K-200K |
| `learning_rate` | Adam optimizer rate | 0.0001-0.001 |
| `gamma` | Discount factor | 0.9-0.99 |
| `batch_size` | Training batch size | 32-128 |

### Network Architecture
| Parameter | Description |
|-----------|-------------|
| `network.type` | `drqn`, `dueling_drqn`, `rainbow_drqn` |
| `network.lstm.hidden_size` | LSTM capacity (128-256) |

### Reward Components
| Parameter | Description |
|-----------|-------------|
| `reward_components.structure.weight` | Structure reward weight |
| `reward_components.transition.weight` | Transition reward weight |
| `reward_components.diversity.weight` | Diversity reward weight |

### Curriculum Learning
| Parameter | Description |
|-----------|-------------|
| `curriculum.enabled` | Enable progressive learning |
| `curriculum.timesteps_per_action` | Phase duration control |
