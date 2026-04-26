# Multi-seed reproducibility runs

Drop training artifacts from additional seeds here. Layout:

```
runs/
├── seed_100/
│   ├── training_scores.json
│   └── training_log.json
├── seed_200/
│   ├── training_scores.json
│   └── training_log.json
└── seed_300/
    ├── training_scores.json
    └── training_log.json
```

The original seed=42 run lives at the repo root (`./training_scores.json`,
`./training_log.json`) — `aggregate_seeds.py --include-default` picks it up.

After dropping artifacts here, run:

```bash
python scripts/aggregate_seeds.py --include-default
```
