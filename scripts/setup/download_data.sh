#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/setup/download_data_%j.out
#SBATCH --error=./logs/setup/download_data_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Download the data
echo "Starting data download..."
python -m beyond_hate.data_processing.download_hateful_meme_hf
