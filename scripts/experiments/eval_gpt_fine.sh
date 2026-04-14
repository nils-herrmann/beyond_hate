#!/bin/bash
#SBATCH --job-name=eval_gpt_fine
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/eval/eval_gpt_fine_%j.out
#SBATCH --error=./logs/eval/eval_gpt_fine_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
echo "Activating beyond_hate conda environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Run GPT-based fine-grained evaluation
echo "Starting GPT-based fine-grained evaluation..."
python -m beyond_hate.eval.eval_fine_gpt
