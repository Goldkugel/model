#!/bin/bash
#SBATCH --job-name=modeltest
#SBATCH --partition=batch_long
#SBATCH --gres=gpu:a40:1                        # Requests 1x A40 GPU
#SBATCH --gpus-per-node=1                       # Forces Slurm to set CUDA_VISIBLE_DEVICES
#SBATCH --export=ALL                            # Passes environment variables into the job
#SBATCH --time=7-00:00:00
#SBATCH --output=../data/slurmlogs/job_%j.out
#SBATCH --error=../data/slurmlogs/job_%j.err

cd ..

python3 ./src/ModelTest.py