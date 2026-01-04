# GHSOM Hierarchy Visualization Notebook - Implementation Summary

**Notebook:** `ghsom_hierarchy_explain.ipynb`
**Purpose:** Academic-quality visualizations for AAMAS conference paper explaining GHSOM hierarchy
**Created:** 2025-12-22

---

## Table of Contents

1. [Overview](#overview)
2. [Model Details](#model-details)
3. [Notebook Structure](#notebook-structure)
4. [Visualization Gallery](#visualization-gallery)
5. [Code Examples](#code-examples)
6. [Output Files](#output-files)
7. [Usage for Academic Publication](#usage-for-academic-publication)

---

## Overview

This notebook provides a comprehensive visual explanation of **Growing Hierarchical Self-Organizing Maps (GHSOM)** using a trained model on the COMMU music dataset. It demonstrates:

- How GHSOM grows and creates hierarchical structure
- Best Matching Unit (BMU) routing through the hierarchy
- Cluster quality and semantic interpretation
- Comparison with flat SOM architecture

### Key Technologies Used

| Component | Library/Tool |
|-----------|--------------|
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| GHSOM Utilities | `ghsom-toolkits` (custom library) |
| Tree Rendering | `pydot`, `graphviz` |
| Quality Metrics | `scikit-learn` |
| Data Processing | `pandas`, `numpy` |

---

## Model Details

### Training Configuration

```yaml
Model Path: experiments/ghsom_commu_full_tsne_optimized_20251125/
Parameters:
  t1 (breadth threshold): 0.35
  t2 (depth threshold): 0.05
  learning_rate: 0.1
  gaussian_sigma: 1.0
  epochs: 30
  decay: 0.99
```

### Resulting Hierarchy

| Metric | Value |
|--------|-------|
| Total Samples | 11,143 |
| Hierarchy Depth | 3 levels |
| Total Maps | 7 |
| Total Neurons | 28 |
| Leaf Clusters | 22 |
| Silhouette Score | ~0.31 |
| Davies-Bouldin Index | ~0.86 |

### Cluster IDs

The 22 leaf clusters have IDs: `[2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]`

---

## Notebook Structure

### Section 1: Title & Abstract
- Model summary table
- Target audience specification

### Section 2: Setup & Configuration
- Publication-quality matplotlib settings
- Nordic color palette definition
- Library imports

```python
# Publication-quality settings applied
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'savefig.dpi': 300,
})

# Nordic color palette
COLORS = {
    'level_0': '#5e81ac',  # Slate blue (root)
    'level_1': '#88c0d0',  # Glacier blue
    'level_2': '#a3be8c',  # Sage green (leaves)
    'highlight': '#bf616a',  # Aurora red
}
```

### Section 3: What is GHSOM? (Conceptual)
- **Figure 1:** 4-panel growth process diagram
- Parameter explanation table

### Section 4: Hierarchy Overview
- **Figure 2:** Full hierarchy tree (Graphviz)
- **Figure 3:** Interactive treemap (Plotly HTML)

### Section 5: BMU Routing
- **Figure 4:** BMU routing examples (3 samples)
- Demonstrates hierarchical path from root to leaf

### Section 6: Node Position Context
- **Figure 5:** Specific node with ancestors/descendants

### Section 7: Weight Analysis
- **Figure 6:** Weight vector heatmap
- **Figure 7:** U-Matrix (cluster boundaries)

### Section 8: Activation Patterns
- **Figure 8:** Activation maps for multiple samples

### Section 9: Cluster Distribution
- **Figure 9:** Histogram + boxplot of cluster sizes
- **Figure 10:** Silhouette plot + quality metrics

### Section 10: Semantic Profiles
- **Figure 11:** 6-panel semantic visualization
  - (a) Communicative functions
  - (b) Dominant musical roles
  - (c) Arousal level pie chart
  - (d) Tempo by cluster
  - (e) Density vs velocity scatter
  - (f) Cohesion score histogram

### Section 11: t-SNE Visualization
- **Figure 12:** 2-panel t-SNE plot
  - (a) Colored by cluster ID
  - (b) Colored by communicative function

### Section 12: GHSOM vs SOM
- **Figure 13:** Conceptual comparison diagram

### Section 13: Summary
- Model statistics recap
- Generated files list
- Figure summary table for paper

---

## Visualization Gallery

### Figure 1: GHSOM Growth Concept
![Growth Concept](../../outputs/ghsom_hierarchy_explain/fig1_ghsom_growth_concept.png)

**Description:** Shows the 4 stages of GHSOM growth:
1. Initial 2x2 map
2. Horizontal growth (when QE > τ₁ × QE₀)
3. Hierarchical expansion (when MQE > τ₂ × QE₀)
4. Final multi-level hierarchy

**Code Example:**
```python
def create_ghsom_growth_concept_figure():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Stage 1: Initial 2x2 map
    for i in range(2):
        for j in range(2):
            circle = plt.Circle((i+0.5, j+0.5), 0.35,
                               color=COLORS['level_0'], ec='black', lw=2)
            axes[0].add_patch(circle)
    # ... continues for stages 2-4
```

---

### Figure 2: Hierarchy Tree
![Hierarchy Tree](../../outputs/ghsom_hierarchy_explain/fig2_ghsom_hierarchy_tree.png)

**Description:** Complete tree structure showing all 28 neurons across 3 levels. Each node displays ID, position, dataset size, and number of children.

**Code Example:**
```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.plotting.hierarchy import visualize_ghsom_hierarchy

# Adapt model for visualization
adapted_model = adapt_model(ghsom_model)
lookup_table = build_lookup_table(adapted_model)

# Generate tree visualization
visualize_ghsom_hierarchy(
    node=adapted_model,
    lookup_table=lookup_table,
    filename="fig2_ghsom_hierarchy_tree.png"
)
```

---

### Figure 4: BMU Routing
![BMU Routing](../../outputs/ghsom_hierarchy_explain/fig4_bmu_routing_sample0.png)

**Description:** Shows how a sample is routed through the hierarchy. Red star marks the sample in t-SNE space, black stars mark BMU at each level.

**Code Example:**
```python
from ghsom_toolkits.interactive import trace_sample_path

# Trace sample through hierarchy
path = trace_sample_path(adapted_model, sample)

# Visualize routing at each level
for step in path:
    level = step['level']
    bmu_pos = step['bmu_position']
    map_shape = step['map_shape']
    has_child = step.get('has_child', False)
```

---

### Figure 11: Semantic Profiles
![Semantic Profiles](../../outputs/ghsom_hierarchy_explain/fig11_cluster_semantic_profiles.png)

**Description:** 6-panel visualization of cluster semantic characteristics from music metadata.

**Code Example:**
```python
def plot_cluster_semantic_profiles(profiles_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Communicative Function Distribution
    func_counts = profiles_df['communicative_function'].value_counts()
    func_counts.plot(kind='barh', ax=axes[0, 0])

    # (b) Dominant Role Distribution
    role_counts = profiles_df['dominant_role'].value_counts()
    role_counts.plot(kind='barh', ax=axes[0, 1])

    # (c) Arousal Level Pie Chart
    arousal_counts = profiles_df['arousal_level'].value_counts()
    axes[0, 2].pie(arousal_counts, autopct='%1.0f%%')

    # ... continues for panels d, e, f
```

---

### Figure 12: t-SNE Clusters
![t-SNE Clusters](../../outputs/ghsom_hierarchy_explain/fig12_tsne_ghsom_clusters.png)

**Description:** 2-panel visualization showing cluster assignments in t-SNE space. Left: colored by cluster ID. Right: colored by communicative function.

**Code Example:**
```python
def plot_tsne_with_ghsom_clusters(data, cluster_ids, profiles_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Color by cluster ID
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        axes[0].scatter(data[mask, 0], data[mask, 1],
                       c=[cmap(i)], s=8, alpha=0.6)

    # Right: Color by communicative function
    for func, color in function_colors.items():
        func_clusters = [c for c, f in cluster_to_function.items() if f == func]
        mask = np.isin(cluster_ids, func_clusters)
        axes[1].scatter(data[mask, 0], data[mask, 1], c=color, label=func)
```

---

### Figure 13: GHSOM vs SOM
![GHSOM vs SOM](../../outputs/ghsom_hierarchy_explain/fig13_ghsom_vs_som.png)

**Description:** Conceptual comparison showing how GHSOM achieves adaptive resolution (7 neurons) vs flat SOM's uniform grid (36 neurons).

---

## Code Examples

### Loading the Model

```python
import pickle
import json
from pathlib import Path

MODEL_DIR = Path("experiments/ghsom_commu_full_tsne_optimized_20251125")

# Load GHSOM model
with open(MODEL_DIR / "ghsom_model.pkl", 'rb') as f:
    ghsom_model = pickle.load(f)

# Load cluster profiles
cluster_profiles = pd.read_csv(MODEL_DIR / "cluster_profiles.csv")

# Load cluster assignments
sample_to_cluster = pd.read_csv(MODEL_DIR / "sample_to_cluster.csv")
cluster_ids = sample_to_cluster['GHSOM_cluster'].values
```

### Adapting Model for Visualization

```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# Adapt to ghsom-toolkits interface
adapted_model = adapt_model(ghsom_model)
lookup_table = build_lookup_table(adapted_model)

print(f"Root map: {adapted_model.rows}x{adapted_model.columns}")
print(f"Total nodes: {len(lookup_table)}")
```

### Computing Quality Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(data, cluster_ids)
db_score = davies_bouldin_score(data, cluster_ids)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
```

### Generating Publication-Quality Figures

```python
# Save figure with publication settings
fig.savefig(
    OUTPUT_DIR / "figure_name.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
)
```

---

## Output Files

### Location
```
outputs/ghsom_hierarchy_explain/
```

### Generated Files

| File | Size | Description |
|------|------|-------------|
| `fig1_ghsom_growth_concept.png` | 357 KB | 4-panel growth diagram |
| `fig2_ghsom_hierarchy_tree.png` | 76 KB | Graphviz tree |
| `fig3_hierarchy_treemap.html` | 4.6 MB | Interactive Plotly treemap |
| `fig4_bmu_routing_sample0.png` | 588 KB | BMU routing example 1 |
| `fig4_bmu_routing_sample5000.png` | 591 KB | BMU routing example 2 |
| `fig4_bmu_routing_sample10000.png` | 503 KB | BMU routing example 3 |
| `fig5_node_position_root.png` | 73 KB | Node context |
| `fig6_weight_heatmap_root.png` | 112 KB | Weight vectors |
| `fig7_umatrix_root.png` | 148 KB | U-Matrix |
| `fig8_activation_maps.png` | 177 KB | Activation patterns |
| `fig9_cluster_distribution.png` | 212 KB | Size distribution |
| `fig10_cluster_quality.png` | 357 KB | Quality metrics |
| `fig11_cluster_semantic_profiles.png` | 588 KB | 6-panel semantics |
| `fig12_tsne_ghsom_clusters.png` | 2.9 MB | t-SNE visualization |
| `fig13_ghsom_vs_som.png` | 647 KB | Comparison diagram |

**Total: 15 files, ~11.6 MB**

---

## Usage for Academic Publication

### Suggested Figure Placement

| Figure | Suggested Section | Purpose |
|--------|-------------------|---------|
| Fig. 1 | Methods | Explain GHSOM algorithm |
| Fig. 2 | Results | Show learned hierarchy |
| Fig. 4 | Methods | Explain BMU routing |
| Fig. 9-10 | Results | Validate cluster quality |
| Fig. 11 | Results | Interpret cluster semantics |
| Fig. 12 | Results | Visualize clustering |
| Fig. 13 | Discussion | Compare with flat SOM |
| Others | Appendix | Supplementary material |

### LaTeX Figure Inclusion

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig1_ghsom_growth_concept.png}
    \caption{GHSOM Growth Process. The algorithm proceeds in stages:
    (1) Initialize a minimal 2$\times$2 map; (2) Grow horizontally when
    quantization error exceeds $\tau_1$ threshold; (3) Expand hierarchically
    when mean quantization error exceeds $\tau_2$ threshold; (4) Continue
    until all neurons satisfy error thresholds.}
    \label{fig:ghsom-growth}
\end{figure}
```

### BibTeX Citation for GHSOM

```bibtex
@article{dittenbach2000growing,
  title={The growing hierarchical self-organizing map},
  author={Dittenbach, Michael and Merkl, Dieter and Rauber, Andreas},
  booktitle={Proceedings of the IEEE-INNS-ENNS International Joint
             Conference on Neural Networks (IJCNN)},
  volume={6},
  pages={15--19},
  year={2000},
  organization={IEEE}
}
```

---

## Dependencies

```
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
plotly>=5.0.0 (optional, for treemap)
pydot>=1.4.0
graphviz (system)
ghsom-toolkits (local package)
```

---

## Quick Start

```python
# Run the notebook
jupyter notebook notebooks/ghsom/ghsom_hierarchy_explain.ipynb

# Or execute via command line
python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('notebooks/ghsom/ghsom_hierarchy_explain.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': 'notebooks/ghsom/'}})

with open('notebooks/ghsom/ghsom_hierarchy_explain.ipynb', 'w') as f:
    nbformat.write(nb, f)
"
```

---

## Contact

For questions about this visualization notebook or the GHSOM model, refer to:
- GHSOM-Toolkits documentation: `/workspace/ghsom-toolkits/README.md`
- Model experiment: `/workspace/experiments/ghsom_commu_full_tsne_optimized_20251125/`

---

*Generated for AAMAS 2025 Conference Submission*
