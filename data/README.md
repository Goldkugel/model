# Data

Output written while running `Model`. The locations of `logs/` and `tmp/` are set in [`config/config.yaml`](../config/config.yaml); the paths below are the defaults.

```
data/
├── logs/       # Run log (logger), prompt log (Model.logPrompts()) and run metadata
└── tmp/        # Generation checkpoint (Model.generate())
```

See [`logs/README.md`](./logs/README.md) and [`tmp/README.md`](./tmp/README.md) for the files in each folder.
