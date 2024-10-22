module load mamba

source activate gw_conda
srun --time=72:00:00 --cpus-per-task=8 --mem-per-cpu=16G --output=logs/generate.log dingo_generate_dataset --settings datasets/waveform_dataset_settings.yaml --out_file datasets/waveform_dataset.hdf5 --num_processes 8
conda deactivate