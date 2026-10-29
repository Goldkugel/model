#!/bin/bash
# Slurm job running the GPU diagnostic in ModelTest.py: checks that CUDA is
# visible and that the first configured model loads. Submit from src/:
#   sbatch runTest.sh
# Paths in the #SBATCH lines are relative to the submission folder (src/).
#SBATCH --job-name=modeltest
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:1                        # Requests 1x A40 GPU
#SBATCH --gpus-per-node=1                       # Forces Slurm to set CUDA_VISIBLE_DEVICES
#SBATCH --export=ALL                            # Passes environment variables into the job
#SBATCH --time=7-00:00:00
#SBATCH --output=../data/slurmlogs/job_%j.out
#SBATCH --error=../data/slurmlogs/job_%j.err

# Run from the repository root, where ./config/config.yaml is found
cd ..

python3 ./src/ModelTest.py