# Data

Output written while running `Model`. The locations of `logs/` and `tmp/` are set in [`config/config.yaml`](../config/config.yaml), the location of `slurmlogs/` in [`scripts/runTest.sh`](../scripts/runTest.sh); the paths below are the defaults.

```
data/
├── logs/       # Run log (logger), prompt log (Model.logPrompts()) and run metadata
├── slurmlogs/  # Output of Slurm jobs (scripts/runTest.sh)
└── tmp/        # Generation checkpoint (Model.generate())
```

See [`logs/README.md`](./logs/README.md), [`slurmlogs/README.md`](./slurmlogs/README.md) and [`tmp/README.md`](./tmp/README.md) for the files in each folder.
