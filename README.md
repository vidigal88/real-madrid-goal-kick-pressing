# Real Madrid Pressing Analysis (Goal Kicks)

End-to-end pipeline for analyzing Real Madrid’s defensive pressure during **opponent goal-kick build-ups** using SkillCorner-style tracking data.

**Pipeline**: Raw tracking → build-up extraction → features → unsupervised pattern discovery → networks/centrality → visualizations + Streamlit viewer

## What you get

- **Extracted build-ups**: ready/kick windows saved as per-build-up parquet files
- **Features**: 20+ per build-up (pressure, compactness, steering, outcomes, QC)
- **Patterns**: GMM zones → tokenization → NMF topics → clustering
- **Networks**: co-pressing partnerships, affinity, centrality, communities
- **Visualizations**: cluster comparisons/sequences, individual trajectories, pressing networks, heatmaps (saved under `visualizations/`)
- **Interactive viewer**: `app.py` (Streamlit) to explore build-ups frame-by-frame

## Data layout (expected)

Place raw data under `data/raw/RealMadrid/`:

```
data/raw/RealMadrid/
├── meta/{game_id}.json
├── dynamic/{game_id}.parquet
└── tracking_parquet/{game_id}.parquet   # preferred
# or: tracking/{game_id}.json            # can be converted and optionally cached
```

## Quick start

### 1) Install (minimal)

This repo currently doesn’t ship pinned `requirements.txt`. Create an environment and install the core packages used by the code:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -U pip
python -m pip install pandas numpy pyarrow scikit-learn scipy networkx matplotlib seaborn plotly streamlit pillow tqdm
```

### 2) Extract build-up windows

Run the extractor from the repo root:

```bash
python src/extract/extraction.py --full --verbose

# or specific matches
python src/extract/extraction.py 2014987 2016604 --verbose

# custom paths
python src/extract/extraction.py --data-root data/raw/RealMadrid --out-dir data/processed/rm_pressing --full
```

This writes:

```
data/processed/rm_pressing/
├── index.parquet
├── params.json
└── frames/build_up_*.parquet
```

### 3) Generate features/models/visualizations

`main.py` runs **feature engineering → modeling → visualization** (it assumes extraction already ran).

```bash
python main.py

# Skip parts if needed
python main.py --skip-models
python main.py --skip-viz
```

Or run steps individually:

```bash
python -m src.features.feature_engineering --processed-root data/processed/rm_pressing --out-dir data/processed/rm_pressing_features --verbose
python -m src.models.gmm_zones
python -m src.models.tokenization
python -m src.models.nmf_topics --n-topics 15
python -m src.models.clustering --n-clusters 5
python -m src.viz.generate_all --out-dir visualizations
```

### 4) Launch the viewer

```bash
streamlit run app.py
```

Defaults:
- processed build-ups: `data/processed/rm_pressing`
- features: `data/processed/rm_pressing_features`
- raw data: `data/raw/RealMadrid`

## Outputs (where to look)

```
data/processed/rm_pressing_features/features.parquet
data/processed/rm_pressing_tokens/
data/processed/rm_pressing_topics/
visualizations/ShortKicks/
visualizations/LongKicks/
```

Notable generated files under each visualization subset:
- `pressure_network_*.png` (full team, frequent players, affinity-weighted, centrality)
- `centrality_metrics.csv`
- `cluster_comparison_*.png`, `cluster_sequence_*.png`
- `individual_{trigger|support1|support2|blocker}_*.png`

## Definitions used in code (important)

There are two related but different concepts:

1) **Frame-level “under pressure”** (features)
- Ball carrier is inferred as the closest opponent to the ball within a radius (`PossessionConfig.possession_radius_m`, default `8.0m`).
- A frame is “under pressure” if the nearest RM defender to the **ball carrier** satisfies either:
  - `dist <= 3.0m`, **or**
  - `dist <= 5.0m` **and** closing speed `>= 1.0 m/s`

See `src/features/services/possession.py` and `src/features/services/pressure.py` (config in `src/features/config.py`).

2) **“Active pressers”** (pressing networks / affinity / centrality)
- For each frame with a detected ball carrier, rank defending players by distance to the carrier.
- Take the **top K closest** (default `K=5`) and count how many frames each defender appears in that top-K set.
- A player is an “active presser” if they appear in top-K for at least `min_frames` (default `10`).

See `src/models/gmm_zones.py` (`identify_pressers`) and `src/models/config.py`.

## Project layout

```
app.py                     # Streamlit viewer
main.py                    # E2E runner (features/models/viz; extraction is separate)
scripts/doc_generator.py   # Docs/code-reference generator
src/extract/               # Extraction (goal-kick build-up windows)
src/features/              # Feature engineering
src/models/                # GMM zones, tokenization, NMF topics, clustering
src/analysis/              # Network centrality + reporting
src/viz/                   # PNG/CSV generation (networks, sequences, trajectories)
docs/                      # Technical docs + interpretation guides
visualizations/            # Example outputs (generated)
```

## Docs

Start here:
- `docs/README.md` (documentation index)
- `docs/pipeline-guide/full_pipeline.md` (end-to-end data flow)
- `docs/visualization-guide/README.md` (how to interpret plots)

## License

MIT (see `LICENSE`).
