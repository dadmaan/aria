# GHSOM Cluster Interpretation Pipeline

A comprehensive analysis pipeline that interprets GHSOM clustering results through the lens of artistic intention, communicative function, and musical preference addressability.

## Scientific Validity Status

| Component | Status | Validation Needed |
|-----------|--------|-------------------|
| Technical implementation | Validated | None |
| Cohesion metrics | Validated | None |
| Categorical aggregation | Validated | None |
| Arousal classification | Heuristic | Listener perception study |
| Role-function mapping | Theoretical | User study (Cohen's kappa > 0.7) |
| Genre-preference mapping | Dataset-dependent | Balanced data collection |

## For Publication

Before submitting for peer review, conduct:

1. **User Validation Study**:
   - Recruit 30+ participants with music background
   - Present cluster samples with role labels
   - Validate communicative function interpretations
   - Compute inter-rater reliability (Cohen's kappa)
   - Target: kappa > 0.7 for substantial agreement

2. **Arousal Perception Study**:
   - Recruit 20+ participants
   - Present cluster samples
   - Collect self-reported arousal ratings (1-7 scale)
   - Correlate with computed arousal scores
   - Target: Pearson r > 0.6

3. **Balanced Data Collection** (ongoing):
   - Collect additional newage/intrinsic samples
   - Target 40-60% genre balance
   - Retrain GHSOM on balanced dataset
   - Regenerate profiles

## Current Use

**Safe for:**
- Exploratory data analysis
- Cluster navigation and visualization
- Hypothesis generation
- Internal experiments

**Not ready for:**
- Claims about listener perception
- Generalizations beyond this dataset

## Overview

This pipeline synthesizes multiple analysis components to transform raw cluster assignments into meaningful interpretations of musical purpose and function. Each cluster is analyzed for its dominant musical characteristics and mapped to communicative intentions.

## Features

- **Intentionality Mapping**: Maps musical roles to communicative functions and artistic intentions
- **Preference Analysis**: Categorizes clusters by intrinsic vs. extrinsic appeal
- **Arousal Classification**: Classifies clusters by emotional arousal level (composite weighted score)
- **Quality Metrics**: Computes clustering quality statistics
- **Validation Framework**: Validates profile-model alignment
- **Comprehensive Reporting**: Generates human-readable reports and visualizations

## Installation

The script requires the following Python packages:
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn (for quality metrics)

Install dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

## Usage

### Basic Usage
```bash
python analysis/ghsom_interpretation/run_interpret_results.py
```

### Custom Configuration
```bash
# Specify custom model directory
python analysis/ghsom_interpretation/run_interpret_results.py --model-dir /path/to/ghsom/output

# Use custom metadata file
python analysis/ghsom_interpretation/run_interpret_results.py --metadata-csv /path/to/features.csv

# Specify output directory
python analysis/ghsom_interpretation/run_interpret_results.py --output-dir /path/to/output

# Use different distance metric
python analysis/ghsom_interpretation/run_interpret_results.py --distance-metric euclidean
```

### Standard Workflow (Current Model)
```bash
python analysis/ghsom_interpretation/run_interpret_results.py \
    --model-dir experiments/ghsom_commu_full_tsne_optimized_20251125 \
    --metadata-csv artifacts/features/raw/commu_full/features_with_metadata.csv \
    --output-dir experiments/ghsom_commu_full_tsne_optimized_20251125
```

### Configuration File
Create a JSON config file:
```json
{
  "model_dir": "/path/to/ghsom/output",
  "metadata_csv": "/path/to/features.csv",
  "output_dir": "/path/to/output",
  "distance_metric": "euclidean"
}
```
Then run:
```bash
python analysis/ghsom_interpretation/run_interpret_results.py --config config.json
```

## Input Requirements

- **GHSOM Model Directory**: Must contain `sample_to_cluster.csv` with cluster assignments
- **Metadata CSV**: Features file with musical metadata (genres, roles, instruments, etc.)
- **Feature Artifact Metadata** (optional): `feature_artifact_metadata.json` for training feature consistency
- **Numeric Features**: Script automatically identifies analyzable numeric columns

## Output Files

The pipeline generates several output files in the specified output directory:

- **`cluster_profiles.csv`**: Detailed profiles for each cluster (22 rows, 21 columns)
- **`interpretation_summary.json`**: Aggregated statistics and quality metrics
- **`interpretation_report.md`**: Human-readable report with cluster interpretations
- **`validation_report.json`**: Profile-model alignment validation results
- **`plots/`**: Directory containing visualization plots:
  - `function_distribution.png`: Distribution of communicative functions
  - `arousal_by_tempo.png`: Cluster size by tempo and arousal level
  - `cohesion_by_role.png`: Cluster cohesion by musical role

## Cluster Interpretation Framework

### Communicative Functions
- **Direct Address**: Music that speaks directly to the listener
- **Narrative Support**: Music that colors and extends the main message
- **Harmonic Foundation**: Music that provides stability and context
- **Atmospheric Creation**: Music that creates space and emotional context
- **Motoric Drive**: Music that propels forward motion and energy

### Preference Types
- **Intrinsic**: Satisfaction from musical structure itself
- **Extrinsic**: Music serving external functions (film, ritual)

### Arousal Classification (Updated)

Arousal is computed using a **composite weighted score**:
- BPM (50% weight): Primary arousal predictor
- Velocity (30% weight): Dynamic intensity
- Density (20% weight): Note activity

Score ranges:
- **Low** (0-33): Calm, contemplative
- **Medium** (33-66): Balanced energy
- **High** (66-100): Intense, driving

## Quality Metrics

- **Silhouette Coefficient**: Measures cluster cohesion vs. separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster scatter (lower is better)
- **Calinski-Harabasz Index**: Ratio of between/within cluster variance (higher is better)

## Key Changes (Phase 1 - December 2025)

1. **Fixed Model Path**: Updated DEFAULT_CONFIG to use current 22-cluster t-SNE model
2. **Feature Space Consistency**: Loads training features (t-SNE) for cohesion metrics
3. **Arousal Classification**: Replaced conjunctive AND logic with composite weighted score
4. **Validation Framework**: Added profile-model alignment validation
5. **Documentation**: Added scientific validity disclaimers

## Troubleshooting

- **File Not Found**: Ensure paths to model directory and metadata CSV are correct
- **No Numeric Features**: Check that metadata contains analyzable numeric columns
- **Quality Metrics Warning**: scikit-learn may not be available; metrics will be set to 0.0
- **Profile-Model Mismatch**: Run validation to check cluster alignment

## Contact

For implementation questions, reference phase files in `docs/MAS-RL/INFERENCE_HIL/GHSOM_PROFILES/PHASES/`.
For scientific validation, consult music psychology experts.

## License

This analysis pipeline is part of the GHSOM music generation research project.
