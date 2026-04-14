from pathlib import Path
import os

import huggingface_hub
import tqdm.auto
from dotenv import load_dotenv
from omegaconf import OmegaConf

from beyond_hate.logger import get_logger

load_dotenv()

# Resolve project root based on this file's location
project_root = Path(__file__).parent.parent.parent.resolve()

# Load configuration
config_path = project_root / 'config/default.yaml'
cfg = OmegaConf.load(config_path)

# Load logger
logs_dir = project_root / cfg.out.logs
logger = get_logger("download_data", logs_dir=logs_dir)


def main():
    """Download the Hateful Memes dataset from Hugging Face."""
    logger.info("Starting data download...")
    
    # Get HF data path from config
    hf_data_path = project_root / cfg.data.paths.hf
    logger.info(f"Download destination: {hf_data_path}")
    
    # Login to HF
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("HF_TOKEN not found in environment variables")
        raise ValueError("HF_TOKEN must be set in .env file")
    
    logger.info("Logging in to Hugging Face...")
    huggingface_hub.login(HF_TOKEN)
    
    # Create dir if does not exist
    logger.info("Creating download directory...")
    hf_data_path.mkdir(parents=True, exist_ok=True)
    
    # Download the dataset from Hugging Face
    logger.info("Downloading dataset from Hugging Face...")
    logger.info("Repository: neuralcatcher/hateful_memes")
    
    huggingface_hub.snapshot_download(
        repo_id="neuralcatcher/hateful_memes",
        repo_type="dataset",
        local_dir=str(hf_data_path),
        max_workers=8,
        force_download=True
    )
    
    logger.info("="*60)
    logger.info("Data download complete!")
    logger.info(f"Dataset saved to: {hf_data_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()