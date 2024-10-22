#!/bin/bash
#SBATCH --job-name=eval_cmpe_tmax10  # Optional: give your job a name
#SBATCH --time=24:00:00            # Time for the job
#SBATCH --gres=gpu:1               # Request one GPU
#SBATCH --constraint=a100          # Constrain to use h100 GPU
#SBATCH --cpus-per-task=16         # Request 48 CPU cores
#SBATCH --mem-per-cpu=8G           # Memory per CPU
#SBATCH --output=training/cmpe_tmax10/logs/eval.log  # Output log file

# Load required module
module load mamba

# Activate conda environment
source activate dingo

# Preload the correct libstdc++ library from conda
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

dingo_pipe training/cmpe_tmax10/GW150914.ini --disable-hdf5-locking=True
# python -m dingo.gw.pipe.sampling training/cmpe_tmax10/GW150914.ini

python eval_plot.py cmpe_tmax10
conda deactivate