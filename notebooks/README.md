# Notebooks

This directory contains Jupyter notebooks for exploration, analysis, and prototyping. Notebooks are organized by topic area.

## ⚠️ Important Notice

**The notebooks in this directory are primarily for exploration and analysis.** For production training and data processing, use the scripts in the `scripts/` directory, which provide more robust, tested, and maintainable implementations.

## Directory Structure

```
notebooks/
├── agents/          # Agent implementations and experiments
├── ghsom/           # GHSOM clustering and analysis
└── preprocessing/   # Data preprocessing explorations
```

## Available Notebooks

### Agents (`agents/`)

#### `10_nb_RLHF_music_environment.ipynb`
**RLHF Music Environment Exploration**

- **Purpose**: Explores the reinforcement learning environment for music generation
- **Status**: Legacy exploration notebook
- **Migration**: Use `src.environments.music_env_gym.MusicGenerationGymEnv` instead
- **See also**: `examples/quickstart_demo.py` for environment usage

**Recommended approach:**
```python
# Modern API (use this)
from src.environments.music_env_gym import MusicGenerationGymEnv
from src.adapters.base import BackendRegistry

adapter = BackendRegistry.create_adapter("sb3")
perceiving_agent = adapter.create_perceiving_agent(config)
env = MusicGenerationGymEnv(perceiving_agent=perceiving_agent, config=config)
```

#### `11_nb_DQN_music_generation.ipynb`
**DQN Music Generation Experiments**

- **Purpose**: Experiments with DQN agent for sequence generation
- **Status**: Legacy exploration notebook
- **Migration**: Use `scripts/run_training.py` for training
- **See also**: `src/adapters/sb3_adapter.py` for modern DQN interface

**Recommended approach:**
```bash
# Modern approach (use this)
python scripts/run_training.py \
    --config configs/agent_config.json \
    --timesteps 10000 \
    --run-id my_experiment
```

#### `12_nb_replay_buffer_analysis.ipynb`
**Replay Buffer Analysis**

- **Purpose**: Analyzes replay buffer contents and sampling strategies
- **Status**: Analysis notebook (still relevant)
- **Note**: This notebook is useful for understanding buffer dynamics
- **API**: Should use `stable_baselines3.common.buffers` for SB3 buffers

#### `13_nb_perceiving_agent.ipynb`
**Perceiving Agent (GHSOM) Exploration**

- **Purpose**: Explores GHSOM-based perceiving agent functionality
- **Status**: Exploratory notebook
- **Migration**: Use `src.agents.agents.PerceivingAgent` for production code
- **See also**: `scripts/train_ghsom.py` for GHSOM training

**Recommended approach:**
```python
# Modern API (use this)
from src.agents.agents import PerceivingAgent

agent = PerceivingAgent(
    ghsom_checkpoint="path/to/checkpoint",
    config=config
)
cluster_ids = agent.evaluate_sequence(features)
action_space = agent.get_action_space()
```

### GHSOM (`ghsom/`)

Notebooks exploring Growing Hierarchical Self-Organizing Maps for music clustering and motif discovery.

**Migration**: Use `scripts/train_ghsom.py` for training, `src/agents/agents.py` for using pretrained models.

### Preprocessing (`preprocessing/`)

Notebooks for data preprocessing experiments.

**Migration**: Use scripts:
- `scripts/fetch_commu_bass.py` - Data fetching
- `scripts/run_feature_extraction_midi.py` - Feature extraction
- `scripts/tsne_reduce_features.py` - Dimensionality reduction

## Using Notebooks

### For Learning and Exploration ✅

Notebooks are excellent for:
- Understanding system components
- Visualizing results and metrics
- Prototyping new ideas
- Interactive debugging
- Analyzing trained models

### For Production/Training ⚠️

**Do NOT use notebooks for:**
- Production training runs (use `scripts/run_training.py`)
- Data pipeline execution (use `scripts/fetch_*.py`, `scripts/run_*.py`)
- Automated workflows (use scripts with proper error handling)
- Reproducible experiments (use versioned configs and scripts)

## Migration Guide

If you have code in notebooks that needs to be production-ready:

### Step 1: Identify the Functionality

Determine what the notebook code does:
- Training? → `scripts/run_training.py`
- Data prep? → `scripts/fetch_*.py` or `scripts/run_*.py`
- Analysis? → Keep in notebook or create new script in `scripts/`

### Step 2: Use Modern APIs

Replace old imports:

```python
# ❌ Old notebook code
from some_old_module import OldClass

# ✅ Modern API
from src.adapters.base import BackendRegistry
from src.agents.agents import PerceivingAgent
from src.environments.music_env_gym import MusicGenerationGymEnv
```

### Step 3: Use Configuration Files

Replace hardcoded values:

```python
# ❌ Old notebook code
learning_rate = 0.001
batch_size = 64
# ... many parameters

# ✅ Modern approach
from src.utils.config_loader import ConfigLoader
config = ConfigLoader("configs/agent_config.json").load()
```

### Step 4: Move to Scripts

For production code:

```python
# ❌ Notebook cells with training logic

# ✅ Create a script or use existing ones
# scripts/run_training.py handles everything
```

## Notebook Best Practices

When working with notebooks in this project:

1. **Add migration notes**: If a notebook uses old APIs, add a cell at the top pointing to the modern equivalent
2. **Keep notebooks simple**: Use them for visualization and exploration, not complex workflows
3. **Import from src/**: Use the production code from `src/` directory
4. **Document assumptions**: Note what data/models the notebook expects
5. **Version control**: Commit notebooks with cleared outputs to avoid large diffs

## Example Migration Notes Template

Add this to the top of notebooks that need migration:

```python
# ⚠️ MIGRATION NOTE
# This notebook uses legacy APIs and is kept for historical reference.
# For production use:
# - Training: python scripts/run_training.py
# - Environment: from src.environments.music_env_gym import MusicGenerationGymEnv
# - Config: configs/agent_config.json
# See README.md for current usage patterns.
```

## Updating Notebooks

To update a notebook to use modern APIs:

1. **Open the notebook**
2. **Add migration note** at the top
3. **Update imports** to use `src.*` modules
4. **Use ConfigLoader** for parameters
5. **Test** that it still runs
6. **Add comments** explaining modern equivalents
7. **Clear outputs** before committing

## Questions?

- See main [README.md](../README.md) for current usage
- Check [CONTEXT.md](../.local/CONTEXT.md) for architecture
- Look at `examples/` for runnable demos
- Refer to `scripts/` for production implementations

## Contributing New Notebooks

When adding new notebooks:

1. Place in the appropriate subdirectory
2. Add a clear title and purpose
3. Use modern APIs from `src/`
4. Include setup instructions
5. Note any prerequisites (data, models)
6. Update this README
7. Clear outputs before committing
