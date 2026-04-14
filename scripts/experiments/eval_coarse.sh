#!/bin/bash
#SBATCH --job-name=eval_coarse
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/eval/eval_coarse_%j.out
#SBATCH --error=./logs/eval/eval_coarse_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Run coarse-grained evaluation
echo "Starting coarse-grained evaluation..."
python -m beyond_hate.eval.eval_coarse