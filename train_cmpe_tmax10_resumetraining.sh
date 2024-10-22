#!/bin/bash
#SBATCH --job-name=cmpe_10_resume  # Optional: give your job a name
#SBATCH --time=120:00:00            # Time for the job
#SBATCH --gres=gpu:1               # Request one GPU
#SBATCH --constraint=a100          # Constrain to use h100 GPU
#SBATCH --cpus-per-task=48         # Request 48 CPU cores
#SBATCH --mem-per-cpu=8G           # Memory per CPU
#SBATCH --output=training/cmpe_tmax10_resumetraining/logs/train.log  # Output log file

# Load required module
module load mamba

# Activate conda environment
source activate dingo

# Preload the correct libstdc++ library from conda
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Run the training command
dingo_train --checkpoint training/cmpe_tmax10_resumetraining/model_400.pt --train_dir training/cmpe_tmax10_resumetraining

# Deactivate the conda environment after completion
conda deactivate
