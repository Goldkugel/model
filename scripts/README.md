# Scripts

| File | Content |
|---|---|
| `runTest.sh` | Slurm job running the GPU diagnostic (`python3 -m model.diagnostic`): checks that CUDA is visible, that a configured model loads and that it generates answers |

Submit it from this folder; the optional argument is the index of the model in `model_id` (default 0):

```bash
sbatch runTest.sh
sbatch runTest.sh 1
```

The job changes to the repository root, where the diagnostic reads `./config/config.yaml`, and writes its output to `data/slurmlogs/`. The checkpoint and run metadata of its test generation go to a temporary folder, whose path it prints. The `model` package must be installed in the job's Python environment.
