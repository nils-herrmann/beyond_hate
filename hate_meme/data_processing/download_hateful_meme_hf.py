import os
from dotenv import load_dotenv

import huggingface_hub
from omegaconf import OmegaConf

load_dotenv()

# Paths
cfg = OmegaConf.load("./config/default.yaml")
hf_data_path = os.path.expanduser(cfg.data.paths.hf)

# Login to HF
HF_TOKEN = os.getenv("HF_TOKEN")
huggingface_hub.login(HF_TOKEN)

# Create dir if does not exist
os.makedirs(hf_data_path, exist_ok=True)
# Download the dataset from Hugging Face
huggingface_hub.snapshot_download(
    repo_id="neuralcatcher/hateful_memes",
    repo_type="dataset",
    local_dir=hf_data_path
)