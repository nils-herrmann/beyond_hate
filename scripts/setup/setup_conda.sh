# Create conda environment
conda create -n beyond_hate \
    python=3.11 \
    pytorch \
    pytorch-cuda=12.1 \
    -c pytorch -c nvidia \
    -y

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beyond_hate

# Install rest of dependencies
pip install -e .
pip install unsloth python-dotenv transformers wandb datasets omegaconf \
           huggingface-hub pandas pillow ipykernel ipywidgets \
           scikit-learn scipy
