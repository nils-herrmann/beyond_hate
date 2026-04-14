#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=./logs/analysis/bias_analysis_%j.out
#SBATCH --error=./logs/analysis/bias_analysis_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Run the bias analysis script
echo "Starting bias analysis..."
python -m beyond_hate.analysis.bias_analysis
