# Scripts Directory

This directory contains utility scripts for the music generation RL pipeline, organized by purpose into functional categories.

## Directory Structure

- **`data_fetching/`** – Dataset acquisition and setup
- **`preprocessing/`** – Feature extraction and dimensionality reduction
- **`ghsom/`** – GHSOM model training, analysis, and visualization
- **`training/`** – RL agent training and inference
- **`demos/`** – Educational demonstrations and tutorials
- **`analysis/`** – Results analysis and comparison
- **`visualization/`** – Publication-quality figures and reports
- **`testing/`** – Validation and testing utilities

## Script Reference

| Script | Category | Purpose |
|--------|----------|---------|
| `fetch_bass_loops.py` | data_fetching | Curate Bass Loops dataset into ComMu-compatible artifacts |
| `fetch_commu_bass.py` | data_fetching | Extract and normalize ComMu bass subset with train/val/test split |
| `combine_datasets.py` | data_fetching | Combine multiple processed datasets into unified metadata |
| `add_filepaths_to_commu_metadata.py` | data_fetching | Add relative file paths to metadata.csv |
| `run_feature_extraction_midi.py` | preprocessing | Extract deterministic features from MIDI files |
| `preprocess_features.py` | preprocessing | Handle missing values and imputation |
| `reduce_features.py` | preprocessing | PCA-based dimensionality reduction |
| `tsne_reduce_features.py` | preprocessing | t-SNE embedding for GHSOM training |
| `filter_dataset_columns.py` | preprocessing | Select and filter CSV columns |
| `run_midi_quantization.py` | preprocessing | Convert audio to MIDI quantization |
| `train_ghsom.py` | ghsom | Train single GHSOM model with exports |
| `run_ghsom_sweep.py` | ghsom | WandB hyperparameter sweep for GHSOM |
| `analyze_ghsom_clusters.py` | ghsom | Comprehensive cluster analysis and characteristics mapping |
| `compute_cluster_quality.py` | ghsom | Evaluate clustering with silhouette, Davies-Bouldin, Calinski-Harabasz metrics |
| `analyze_tsne_dataset.py` | ghsom | Dataset analysis for GHSOM training |
| `verify_ghsom_artifacts.py` | ghsom | Validate GHSOM checkpoint and embedding files |
| `visualize_ghsom_model.py` | ghsom | Generate GHSOM hierarchy and U-Matrix visualizations |
| `run_training.py` | training | Main RL training entry point (SB3 backend) |
| `train_music_generation_sb3.py` | training | SB3 DQN agent training with callbacks |
| `run_inference.py` | training | Generate music from trained agent |
| `pretrain_lstm.py` | training | Pretrain LSTM on cluster sequences |
| `pretrain_lstm_tf.py` | training | TensorFlow LSTM pretraining |
| `run_t1_sweep.py` | training | Streamlined t1 hyperparameter tuning |
| `demo_composite_reward_shaping.py` | demos | Composite reward system with human feedback |
| `demo_lstm_enhanced_dqn.py` | demos | LSTM-enhanced DQN for temporal modeling |
| `demo_multiagent_workflow.py` | demos | Multi-agent architecture walkthrough |
| `demo_multibackend_rl_framework.py` | demos | SB3 and TF-Agents backend comparison |
| `demo_sb3_tfa_backend_parity.py` | demos | Backend compatibility and consistency |
| `demo_test_suite_showcase.py` | demos | Testing framework and coverage metrics |
| `analyze_training_metrics.py` | analysis | Learning curves and convergence analysis |
| `compare_baseline_policies.py` | analysis | Statistical comparison vs. random/uniform policies |
| `compare_quality_metrics.py` | analysis | Cross-feature-space quality comparison |
| `compare_t1_results.py` | analysis | Hyperparameter sweep results summary |
| `benchmark_computational_performance.py` | analysis | Training time, memory, latency profiling |
| `run_ablation_studies.py` | analysis | Component removal experiments |
| `visualize_ghsom.py` | visualization | GHSOM hierarchy and weight heatmaps |
| `visualize_t1_comparison.py` | visualization | t1 parameter sensitivity plots |
| `visualize_comprehensive_results.py` | visualization | Publication-ready figures and comparisons |
| `create_hierarchy_tree_viz.py` | visualization | Custom hierarchy tree rendering |
| `create_summary_dashboard.py` | visualization | Comprehensive summary dashboard |
| `generate_paper_tables.py` | visualization | LaTeX tables for publication |
| `final_validation.py` | testing | Final validation of implementation |
| `test_common_root_implementation.py` | testing | Path resolution testing |
| `test_e2e_common_root.py` | testing | End-to-end integration tests |

## Usage Examples

```bash
# Data pipeline: fetch → preprocess → reduce
python scripts/data_fetching/fetch_commu_bass.py --source-dir data/raw/commu
python scripts/preprocessing/run_feature_extraction_midi.py --dataset-root data/raw/commu_bass
python scripts/preprocessing/tsne_reduce_features.py --input-features artifacts/features/raw/commu_bass

# Model training: GHSOM → RL agent
python scripts/ghsom/train_ghsom.py --feature-path artifacts/features/tsne/commu_bass_tsne
python scripts/training/run_training.py --config configs/agent_config.json

# Analysis and visualization
python scripts/analysis/compare_baseline_policies.py --model-path outputs/run_*/checkpoints/final
python scripts/visualization/visualize_comprehensive_results.py --training-dir analysis/training_metrics
```

## Adding New Scripts

When creating new scripts, place them in the most appropriate directory based on function:
- **Phase correlation** – Does it fit an existing pipeline phase (fetch/preprocess/train/analyze)?
- **Primary use case** – Is it primarily for demos, testing, or production?
- **Fallback** – When in doubt, demos > analysis > testing

If no category fits, consider creating a new subdirectory with a clear purpose.

# 2. Run GHSOM hyperparameter sweep
python scripts/run_ghsom_sweep.py --config configs/ghsom_sweep.json

# 3. Pretrain LSTM models
python scripts/pretrain_lstm.py --data data/processed/sequences.csv --output models/lstm_pytorch
python scripts/pretrain_lstm_tf.py --data data/processed/sequences.csv --output models/lstm_tensorflow

# 4. Train RL music generation agent
python scripts/train_music_generation_sb3.py --config configs/agent_config.json --timesteps 10000
```

### Testing & Demonstration
```bash
# Test backend integration and interface compliance
python scripts/demo_backend_integration.py --backend test

# Run SB3 backend demo with verbose output
python scripts/demo_backend_integration.py --backend sb3 --episodes 3 --verbose

# Run TF-Agents backend demo
python scripts/demo_backend_integration.py --backend tfa --episodes 2

# Run comprehensive integration tests
python -m pytest test/test_backend_adapters_integration.py -v

# Run composite reward shaping demo
python scripts/demo_composite_reward_shaping.py
```

### Composite Reward Shaping Demo
The `demo_composite_reward_shaping.py` script demonstrates the complete reward shaping system:
- **Similarity metrics**: Shows cosine, euclidean, token equality, and attribute-level comparisons
- **Human feedback**: Demonstrates timeout handling, non-interactive mode, and CLI integration
- **Reward normalization**: Shows robust handling of edge cases (NaN, inf, zero vectors)
- **Formula compliance**: Validates CONTEXT.md formula `R_total = w1*R_sim + w2*R_str + w3*R_hum`
- **Full integration**: End-to-end episode reward calculation with all components

```bash
# Run with default configuration
python scripts/demo_composite_reward_shaping.py

# The demo creates a configuration file for reference:
# scripts/composite_reward_demo_config.json
```

## Configuration

### Main Configuration Files
- `configs/agent_config.json` - Main RL agent configuration used by training and demo scripts
- Additional config files may be used by specific preprocessing scripts

### Key Configuration Settings (agent_config.json)
- `rl_backend`: Choose between 'sb3' (Stable-Baselines3) or 'tfa' (TF-Agents)
- `reward_weights`: Configure composite reward components (w1: similarity, w2: structure, w3: human)
- `similarity_mode`: Feature similarity calculation method ('cosine', 'euclidean', 'attribute')
- `enable_wandb`: Enable/disable Weights & Biases experiment tracking
- `enable_gpu`: Enable/disable GPU acceleration
- `sequence_length`: Length of generated music sequences
- `learning_rate`: Learning rate for RL agent training
- `batch_size`: Batch size for training
- `gamma`: Discount factor for future rewards

### Script-Specific Arguments
Most scripts accept command-line arguments for input/output paths, hyperparameters, and execution options. Use `--help` with any script to see available options:

```bash
python scripts/[script_name].py --help
```

## Typical Workflow

The scripts are designed to work together in a data processing and training pipeline:

1. **Data Collection**: Use fetch scripts to gather raw MIDI data
2. **Preprocessing**: Extract features and quantize MIDI files  
3. **Dimensionality Reduction**: Apply PCA/t-SNE for visualization and efficiency
4. **Structural Analysis**: Train GHSOM to discover musical patterns and clusters
5. **Sequence Modeling**: Pretrain LSTM models for temporal sequence understanding
6. **RL Training**: Train generative agents using the pretrained components
7. **Evaluation**: Use demo scripts to test and validate trained models

## Dependencies

Scripts require different combinations of:
- **Core**: numpy, pandas, scikit-learn  
- **Music Processing**: music21, pretty_midi, muspy, note_seq
- **Deep Learning**: torch, tensorflow
- **RL Frameworks**: stable-baselines3, tf-agents
- **Visualization**: matplotlib, seaborn, plotly
- **Clustering**: networkx (for GHSOM)
- **Utilities**: jsonschema (for config validation)

See `requirements.txt` for complete dependency list with specific versions.

## Notes

- Most scripts support both CPU and GPU execution where applicable
- Scripts follow the unified configuration pattern and respect global settings
- All scripts include `--help` documentation for their specific parameters
- Output paths are typically configurable and will create directories as needed