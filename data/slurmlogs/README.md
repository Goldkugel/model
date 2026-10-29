# Slurm Logs

Output of the Slurm jobs in [`src/`](../../src/), e.g. `runTest.sh`:

| File | Content |
|---|---|
| `job_<id>.out` | Standard output of job `<id>` |
| `job_<id>.err` | Standard error of job `<id>` |

Slurm does not create this folder, so it is kept in git through this README; every other file in it is ignored by git.
