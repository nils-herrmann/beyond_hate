#!/bin/bash
#SBATCH --job-name=train_coarse
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/train/train_coarse_%j.out
#SBATCH --error=./logs/train/train_coarse_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Run coarse-grained training
echo "Starting coarse-grained training..."
python -m beyond_hate.train.train_coarse