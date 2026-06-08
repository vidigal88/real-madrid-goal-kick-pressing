# Tutorial Notebooks

This directory contains interactive Jupyter notebooks demonstrating the Real Madrid Pressing Analysis workflow.

## Available Notebooks

### 1. Extraction Walkthrough ([01_extraction_walkthrough.ipynb](01_extraction_walkthrough.ipynb))
Learn how to extract goal-kick build-up windows from SkillCorner tracking data.

**Topics Covered:**
- Loading match tracking data
- Detecting goal kick events
- Extracting time windows around kicks
- Understanding the extraction pipeline

**Duration:** ~15 minutes

---

### 2. Feature Analysis ([02_feature_analysis.ipynb](02_feature_analysis.ipynb))
Explore feature engineering for pressing analysis.

**Topics Covered:**
- Coordinate normalization
- Ball carrier inference
- Pressure metrics computation
- Team compactness features
- Quality control checks

**Duration:** ~20 minutes

---

### 3. Model Training ([03_model_training.ipynb](03_model_training.ipynb))
Train machine learning models to discover pressing patterns.

**Topics Covered:**
- GMM zone learning
- Tokenization of player movements
- NMF topic discovery
- Hierarchical clustering

**Duration:** ~25 minutes

---

### 4. Visualization Gallery ([04_visualization_gallery.ipynb](04_visualization_gallery.ipynb))
Create compelling visualizations of pressing patterns.

**Topics Covered:**
- Pressing heatmaps
- Player trajectories
- Convex hull team shapes
- Pressing networks

**Duration:** ~20 minutes

---

### 5. Network Analysis ([05_network_analysis.ipynb](05_network_analysis.ipynb))
Analyze pressing partnerships using graph theory.

**Topics Covered:**
- Co-pressing networks
- Centrality metrics (degree, betweenness)
- Community detection
- Key player identification

**Duration:** ~25 minutes

---

## Getting Started

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook

# Install project dependencies
pip install -r requirements.txt
```

### Running Notebooks

```bash
# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to this `notebooks/` directory and open any `.ipynb` file.

### Data Requirements

Most notebooks require:
- Extracted build-up data in `data/processed/rm_pressing/`
- Sample tracking data from SkillCorner

See [Data Preparation Guide](../docs/getting-started/data-preparation.md) for setup instructions.

---

## Notebook Order

For first-time users, follow this recommended sequence:

1. **Extraction** → Understand data extraction
2. **Features** → Learn feature engineering
3. **Models** → Discover patterns with ML
4. **Visualization** → Create compelling graphics
5. **Network** → Analyze player partnerships

---

## Additional Files

- **Metadata_Analysis.ipynb** - Legacy notebook for exploring metadata structure
- **topic_analysis.py** - Python script for batch topic analysis

---

## Tips for Learning

- **Run cells sequentially** - Each cell builds on previous computations
- **Modify parameters** - Experiment with different settings
- **Visualize results** - Charts help understand data patterns
- **Read docstrings** - Hover over functions to see documentation

---

## Troubleshooting

**Issue:** Missing data files
**Solution:** Run extraction pipeline first (see [Quick Start](../docs/getting-started/quick-start.md))

**Issue:** Import errors
**Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue:** Kernel crashes
**Solution:** Restart kernel and reduce data sample size

---

## Contributing

Found an issue or want to add a notebook? See [Contributing Guide](../docs/development/contributing.md).

---

## See Also

- [Documentation Home](../docs/index.md)
- [API Reference](../docs/api-reference/)
- [Concept Guides](../docs/concepts/)
