# Real Madrid Goal-Kick Pressing

Football analytics project studying how Real Madrid pressed opponents after goal-kick restarts across two seasons.

This repository is a follow-up to an initial Soccermatics course project developed with two colleagues under Professor David Sumpter. The original version was a first prototype. I later continued the goal-kick detection, feature engineering, clustering, and visualisation work independently to create a larger and more robust analysis.

## Project Summary

The analysis focuses on **opponent goal kicks against Real Madrid** and asks:

- How often do opponents play short or direct from goal kicks?
- What pressing structures does Real Madrid use against short restarts?
- Are the same pressing patterns visible across seasons?
- Can unsupervised learning reveal recurring team behaviours from tracking data?

The updated pipeline detected **552 opponent goal-kick build-ups** across two seasons.

Current restart split:

```text
Short restarts: 486
Direct restarts: 66
```

For the public tactical analysis, the focus is on short restarts, defined as goal kicks where the first reception is within **15 metres** of the kick.

## Key Findings

For short opponent goal kicks, the cluster sequence plots suggest two main Real Madrid pressing behaviours:

- **Cluster 0: controlled high containment**  
  Real Madrid are already positioned high near the opponent penalty area, but the team shape is more stretched. The first seconds after the kick show structural stability rather than an immediate aggressive jump.

- **Cluster 1: compact high-pressure trap**  
  Real Madrid are more compact around the ball-side build-up zone, creating a clearer high-intensity trap near the opponent box.

These two behaviours appear in both `2023/2024` and `2024/2025`, suggesting continuity in Real Madrid's pressing approach despite changes in personnel.

Short-restart cluster sizes:

```text
2023/2024
Cluster 0: 125
Cluster 1: 111
Cluster 2: 3
Cluster 3: 3

2024/2025
Cluster 0: 130
Cluster 1: 112
Cluster 2: 2
```

Clusters with very small samples are treated as outliers rather than tactical families.

## Methodology

### Goal-Kick Detection

Goal kicks were detected using a combination of event data and tracking validation.

The event data first identifies candidate moments where the opponent has a `goal_kick_for` restart. Tracking data is then used to validate the scene and kick moment:

- Ball located in or very near the opponent goal area.
- Kick-like ball movement detected through speed, acceleration, and future displacement.
- Real Madrid players not repeatedly inside the opponent goal area at the restart moment.
- Opponent goalkeeper context checked through goalkeeper-ball proximity.
- Duplicate references to the same kick time removed.

This approach improved the sample from the original prototype to **552 detected build-ups**.

### Restart Classification

Restarts are classified by first-reception distance:

```text
Short:  first reception < 15m
Direct: first reception >= 15m
```

True long restarts, defined as `>= 30m`, are retained as a descriptive sub-label but are rare in this dataset.

### Pressing Pattern Discovery

The clustering is based on Real Madrid's pressing movements, not manual labels.

Pipeline:

1. Normalize pitch direction so the opponent always builds from left to right.
2. Identify active Real Madrid pressers based on repeated proximity to the ball carrier.
3. Sample presser positions around `kick + 1s` and `kick + 5s`.
4. Map those positions into data-driven pitch zones using Gaussian Mixture Models.
5. Convert each pressing movement into a zone-transition token.
6. Build a token matrix representing each goal-kick build-up.
7. Apply NMF topic modelling to identify recurring pressing motifs.
8. Cluster build-ups by their NMF topic profiles.

Model settings used in the current version:

```text
Goal-kick build-ups: 552
Movement tokens: 120
NMF topics: 8
Clusters: 4
Silhouette score: 0.229
```

The decision to use **8 topics** and **4 clusters** prioritised interpretability and tactical readability over creating many small groups.

## Visualisations

The main public visuals are the `cluster_sequence_*` plots for short restarts, split by season.

Each sequence plot shows:

- Real Madrid average player positions.
- Opponent average player positions.
- Average ball position.
- Team convex hulls.
- The number of goal kicks contributing to each timestamp.

Important interpretation note:

> Player and ball positions are cluster averages at each timestamp.

The first frame is labelled **Goal-kick setup**. Subsequent frames show `Kick + 2s`, `Kick + 4s`, and so on. Later timestamps may use fewer goal kicks because not every saved tracking window extends to `+8s` or `+10s`.

## Repository Structure

```text
main.py                    # Pipeline runner for features/models/visualisations
src/extract/               # Goal-kick detection and build-up extraction
src/features/              # Feature engineering
src/models/                # GMM zones, tokenisation, NMF topics, clustering
src/analysis/              # Analysis helpers
src/viz/                   # Plot generation
notebooks/                 # Exploratory notebooks
visualizations/            # Generated public visual outputs
```

## Data Availability

The raw and processed tracking/event data are **not included** in this repository due to data licensing restrictions.

The code is provided to document the methodology and analysis workflow. To reproduce the full pipeline, compatible event and tracking data with the expected structure are required.

Expected local data structure:

```text
data/raw/RealMadrid/
├── meta/{game_id}.json
├── dynamic/{game_id}.parquet
└── tracking_parquet/{game_id}.parquet
```

The `data/` folder is intentionally excluded from version control.

## Example Commands

If compatible data is available locally, the main steps are:

```bash
python src/extract/extraction.py --full --verbose

python -m src.features.feature_engineering \
  --processed-root data/processed/rm_pressing \
  --out-dir data/processed/rm_pressing_features \
  --verbose

python -m src.models.gmm_zones
python -m src.models.tokenization
python -m src.models.nmf_topics --n-topics 8
python -m src.models.clustering --n-clusters 4
python -m src.viz.generate_all --out-dir visualizations
```

## Notes

This is an ongoing project. The current public version focuses on team-level pressing structures from short goal kicks. Player-role analysis, such as first presser, support presser, and channel blocker identification, is intentionally left for a future phase.

## License

MIT License. See `LICENSE`.
