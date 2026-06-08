# Data Folder Template

This folder documents the expected local data layout without including any real data.

The actual `data/` folder is intentionally excluded from version control because the raw and processed event/tracking data are subject to licensing restrictions.

To run the full pipeline locally, create the following private folder structure:

```text
data/
├── raw/
│   └── RealMadrid/
│       ├── meta/
│       │   └── {game_id}.json
│       ├── dynamic/
│       │   └── {game_id}.parquet
│       └── tracking_parquet/
│           └── {game_id}.parquet
├── interim/
└── processed/
```

Do not commit files from `data/`, including derived `.parquet`, `.pkl`, `.json`, or compressed archives.
