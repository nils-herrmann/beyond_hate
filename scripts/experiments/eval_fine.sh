#!/bin/bash
#SBATCH --job-name=eval_fine
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/eval/eval_fine_%j.out
#SBATCH --error=./logs/eval/eval_fine_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Run fine-grained evaluation
echo "Starting fine-grained evaluation..."
python -m beyond_hate.eval.eval_fine