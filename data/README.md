# Data

Output written while running `Model`. The locations of `logs/` and `tmp/` are set in [`config/config.yaml`](../config/config.yaml), the location of `slurmlogs/` in [`src/runTest.sh`](../src/runTest.sh); the paths below are the defaults.

```
data/
├── logs/       # Run log (logger) and prompt log (Model.logPrompts())
├── slurmlogs/  # Output of Slurm jobs (src/runTest.sh)
└── tmp/        # Generation checkpoint (Model.generate())
```

See [`logs/README.md`](./logs/README.md), [`slurmlogs/README.md`](./slurmlogs/README.md) and [`tmp/README.md`](./tmp/README.md) for the files in each folder.
