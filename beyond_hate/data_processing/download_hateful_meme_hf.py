import os
import huggingface_hub
from dotenv import load_dotenv

load_dotenv()

# Paths
hf_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "hateful_memes_hf"
)

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