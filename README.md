# ARIA: Multi-Agent Music Generation with Reinforcement Learning

**ARIA**: **A**utonomous **R**einforcement-learning with **I**ntelligent **A**bstraction

A multi-agent reinforcement learning framework for symbolic music generation with human-in-the-loop adaptation.

## Overview

This project implements a modular RL framework combining:

- **Perceiving Agent (GHSOM)**: Clusters musical features and discovers structural motifs using Growing Hierarchical Self-Organizing Maps
- **Generative Agent (DQN)**: Selects musical elements using recurrent Q-learning with Tianshou 2.0
- **Human Agent**: Provides multi-dimensional feedback for preference-guided adaptation

The system uses **Tianshou 2.0** for reinforcement learning with support for multiple algorithms, curriculum learning, and human-in-the-loop preference adaptation.

## Quickstart

### Docker (Recommended)

Docker provides an isolated environment with CUDA 12.1 + cuDNN 8 for GPU acceleration.

**Prerequisites:**
- Docker Engine 20.10+ and Docker Compose v2.0+
- NVIDIA Container Toolkit for GPU support ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

```bash
# Build the image
cd docker && docker-compose build

# Start container
docker-compose up -d

# Access container shell
docker-compose exec music-generation-rl bash

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run training (default command)
python scripts/training/run_training.py --config configs/training.yaml
```

**Jupyter Notebook:**
```bash
docker-compose --profile notebook up -d
# Access at http://localhost:8888
```

**Container Management:**
```bash
docker-compose down          # Stop containers
docker-compose logs -f       # View logs
docker-compose restart       # Restart after code changes
```

### Local Installation

```bash
# Python 3.10+
pip install -r requirements.txt
```

> **Dependencies:** Tianshou 2.0, PyTorch 2.0+, Gymnasium 1.0+. GHSOM packages are installed from local `ghsom-py/` and `ghsom-toolkits/` directories.

## Algorithms

ARIA supports multiple DQN algorithm variants via Tianshou 2.0:

| Algorithm | Description |
|-----------|-------------|
| `dqn` | Double DQN with target network |
| `dueling_dqn` | Separate value/advantage streams |
| `c51` | Categorical distributional RL (51 atoms) |
| `rainbow` | Full Rainbow (Dueling + C51 + NoisyNet + PER) |

Select algorithm in `configs/training.yaml`:
```yaml
algorithm:
  type: "rainbow"  # dqn, dueling_dqn, c51, rainbow
  is_double: true
```

## Training

```bash
# Train with default configuration
python scripts/training/run_training.py

# Train with custom settings
python scripts/training/run_training.py \
    --config configs/training.yaml \
    --timesteps 200000 \
    --seed 42

# Train with specific algorithm config
python scripts/training/run_training.py \
    --config configs/benchmark/main/rainbow_cl.yaml
```

**Output Structure:**
```
outputs/run_YYYYMMDD_HHMMSS/
├── checkpoints/     # Model checkpoints
├── logs/            # TensorBoard logs
├── metrics/         # Training metrics (JSON)
└── configs/         # Saved configuration
```

## Inference

```bash
# Interactive HIL session
python scripts/inference/run_inference_pipeline.py hil \
    --checkpoint artifacts/training/run_xyz/checkpoints/final.pth

# Batch generation
python scripts/inference/run_inference_pipeline.py analyze \
    --sequences outputs/inference/sequences.json

# Preference-guided simulation
python scripts/inference/run_inference_pipeline.py simulate \
    --checkpoint path/to/checkpoint.pth \
    --scenario calm_relaxation
```

Available subcommands: `hil`, `analyze`, `benchmark`, `paper`, `simulate`, `visualize-simulation`

## Human-in-the-Loop (HITL)

ARIA supports preference-guided adaptation with multi-dimensional feedback:

- **Feedback dimensions**: Quality, Coherence, Creativity, Musicality (1-5 scale)
- **Adaptation modes**: `q_penalty` (fast behavioral adjustment) or `reward_shaping` (gradual learning)
- **Pre-defined scenarios**: `calm_relaxation`, `energetic_drive`, `piano_focus`, `melodic_focus`

See `configs/inference.yaml` for detailed HITL configuration options.

## Configuration

Configuration uses hierarchical YAML files:

**Training (`configs/training.yaml`):**
```yaml
algorithm:
  type: "dqn"
  is_double: true

network:
  type: "drqn"           # drqn, mlp, dueling_drqn, c51_drqn, rainbow_drqn
  lstm:
    hidden_size: 256

training:
  total_timesteps: 200000
  learning_rate: 0.0001
  batch_size: 64

curriculum:
  enabled: true
```

**Inference (`configs/inference.yaml`):**
```yaml
adaptation:
  default_mode: "reward_shaping"
  reward_shaping:
    strength: 10.0
    decay_factor: 0.98
```

## Project Structure

```
aria/
├── src/
│   ├── adapters/           # Tianshou adapter
│   ├── agents/             # GHSOM perceiving agent
│   ├── curriculum/         # Curriculum learning callbacks
│   ├── environments/       # Gymnasium-compliant music env
│   ├── inference/          # Interactive & preference-guided sessions
│   ├── networks/           # Q-networks (DRQN, MLP, Dueling, C51, Rainbow)
│   ├── training/           # Tianshou trainer
│   └── utils/              # Logging, rewards, MIDI utilities
├── scripts/
│   ├── training/           # run_training.py
│   ├── inference/          # run_inference_pipeline.py
│   └── benchmark/          # Ablation studies
├── configs/
│   ├── training.yaml
│   ├── inference.yaml
│   └── benchmark/          # 16+ config variants
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
└── test/                   # Unit tests
```

## Testing & Development

```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Lint
ruff check src/ scripts/

# Format
black src/ scripts/
```

## Troubleshooting

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Ensure NVIDIA drivers and CUDA toolkit are installed.

**Docker GPU issues:**
Verify NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Training hangs:**
Set `non_interactive_mode: true` in config to bypass human input prompts.

**WandB prompts:**
Set `enable_wandb: false` in config or `WANDB_MODE=disabled` environment variable.

## References

- [Tianshou Documentation](https://tianshou.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome via pull requests:
1. Follow PEP 8
2. Add tests
3. Ensure tests pass
