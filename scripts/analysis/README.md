# Analysis Scripts

This directory contains analysis scripts organized by their function in the analysis workflow.

## Directory Structure

```
scripts/analysis/
├── 01_data_exploration/       # Initial dataset exploration and visualization
├── 02_feature_importance/     # Feature importance analysis (SHAP, RFE, VIF)
├── 03_cluster_interpretation/ # Interpret GHSOM clusters semantically
├── 04_cluster_validation/     # Prepare clusters for listening tests
├── 05_embedding_visualization/# Visualize clusters in t-SNE space
├── 06_training_analysis/      # Analyze and compare training runs
└── results/                   # Training run output visualizations
```

## Workflow Overview

The analysis pipeline follows this general order:

### 1. Data Exploration (`01_data_exploration/`)

Start here to understand your dataset characteristics and validate t-SNE embedding quality.

| Script | Description |
|--------|-------------|
| `explore_dataset.py` | Comprehensive EDA with visualizations for genre, BPM, pitch range, time signatures, and data quality |
| `visualize_tsne_embeddings.py` | t-SNE embedding visualization with automatic K-means clustering |
| `compare_feature_set_tsne.py` | Compare t-SNE quality between Full 33D vs Filtered 17D feature sets |
| `analyze_tsne_stability.py` | Comprehensive t-SNE stability analysis: seed robustness, perplexity sensitivity, embedding quality metrics |

**Output:** Distribution plots, correlation matrices, t-SNE visualizations, stability analysis reports (ARI/NMI heatmaps, perplexity curves, trustworthiness plots)

### 2. Feature Importance (`02_feature_importance/`)

Analyze which features matter most for clustering.

| Script | Description |
|--------|-------------|
| `scripts/run_full_pipeline.py` | Run complete analysis: EDA → RFE → SHAP → Consensus |
| `scripts/run_quick_analysis.py` | Quick feature importance ranking |
| `scripts/feature_distributions.py` | Feature distribution analysis |
| `scripts/rfe_analysis.py` | Recursive Feature Elimination |
| `scripts/shap_analysis.py` | SHAP-based interpretability |
| `scripts/vif_analysis.py` | Multicollinearity detection (VIF) |
| `consensus_analysis.py` | Combine multiple methods into consensus ranking |

**Output:** Feature rankings, SHAP plots, consensus reports

### 3. Cluster Interpretation (`03_cluster_interpretation/`)

Understand what each GHSOM cluster represents musically.

| Script | Description |
|--------|-------------|
| `interpret_clusters.py` | Generate cluster profiles, intentionality mapping, and validation reports |

**Output:** `cluster_profiles.csv`, `cluster_interpretation_report.md`, `intentionality_mapping.json`

### 4. Cluster Validation (`04_cluster_validation/`)

Prepare cluster samples for qualitative evaluation (listening tests).

| Script | Description |
|--------|-------------|
| `scripts/01_select_samples.py` | Select representative samples from each cluster |
| `scripts/02_render_audio.py` | Convert MIDI samples to audio (WAV/MP3) |
| `scripts/03_analyze_similarity.py` | Analyze intra-cluster similarity metrics |
| `scripts/run_pipeline.py` | Run the complete validation pipeline |

**Output:** Sampled audio files, similarity analysis reports

### 5. Embedding Visualization (`05_embedding_visualization/`)

Visualize GHSOM clusters projected onto t-SNE embeddings.

| Script | Description |
|--------|-------------|
| `visualize_cluster_embeddings.py` | Combine t-SNE embeddings with cluster labels, compute quality metrics |

**Output:** 2D cluster visualizations, silhouette/Davies-Bouldin scores

### 6. Training Analysis (`06_training_analysis/`)

Analyze and compare RL training runs.

| Script | Description |
|--------|-------------|
| `analyze_training_runs.py` | Comprehensive training analysis with metrics and comparisons |
| `run_discovery.py` | Discover and catalog training runs |
| `generate_paper_tables.py` | Generate publication-ready LaTeX tables |
| `data_loaders.py` | Load training configs, metrics, TensorBoard data |
| `latex_tables.py` | LaTeX table generation utilities |

**Output:** Learning curves, convergence analysis, markdown/LaTeX reports

### Results (`results/`)

Contains output visualizations from completed training runs:
- Component evolution plots
- Episode rewards
- Visitation heatmaps
- Hierarchy navigation visualizations

## Quick Start

```bash
# 1. Explore your dataset
python scripts/analysis/01_data_exploration/explore_dataset.py \
    --metadata_path artifacts/features/metadata.csv \
    --output_dir outputs/eda

# 1b. Analyze t-SNE stability (optional, comprehensive validation)
python scripts/analysis/01_data_exploration/analyze_tsne_stability.py \
    --output-dir outputs/tsne_stability

# 2. Analyze feature importance
python scripts/analysis/02_feature_importance/scripts/run_full_pipeline.py \
    --data_path artifacts/features/features_numeric.csv \
    --output_dir outputs/feature_analysis

# 3. Interpret clusters
python scripts/analysis/03_cluster_interpretation/interpret_clusters.py \
    --cluster_assignments artifacts/ghsom/sample_to_cluster.csv \
    --output_dir outputs/interpretation

# 4. Analyze training runs
python scripts/analysis/06_training_analysis/analyze_training_runs.py \
    --input-dir artifacts/training \
    --output-dir outputs/training_analysis
```

## Dependencies

Most scripts require:
- pandas, numpy, matplotlib, seaborn
- scikit-learn (for clustering metrics, ARI/NMI, trustworthiness)
- shap (for feature importance)
- tensorboard (for training analysis)

See `requirements.txt` in the project root for the complete list.
