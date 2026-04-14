from pathlib import Path
import os

import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf

from beyond_hate.logger import get_logger

# Resolve project root based on this file's location
project_root = Path(__file__).parent.parent.parent.resolve()

# Load configuration
config_path = project_root / 'config/default.yaml'
cfg = OmegaConf.load(config_path)

# Load logger
logs_dir = project_root / cfg.out.logs
logger = get_logger("descriptive_stats", logs_dir=logs_dir)


def load_full_dataset():
    """Load the full Hateful Memes dataset from HuggingFace files."""
    logger.info("Loading full Hateful Memes dataset...")
    
    hf_data = project_root / cfg.data.paths.hf
    
    # Load all splits
    df_full = pd.DataFrame()
    splits = ['dev_seen', 'dev_unseen', 'test_seen', 'test_unseen', 'train']
    
    for split in splits:
        split_file = hf_data / f'{split}.jsonl'
        df_tmp = pd.read_json(split_file, lines=True)
        df_tmp['split'] = split
        df_full = pd.concat([df_full, df_tmp])
    
    logger.info(f"Total number of labels: {len(df_full)}")
    
    # Verify if images exist in the hf_data directory
    hf_images = os.listdir(hf_data / "img")
    df_full['image_found'] = df_full['img'].str.lstrip('img/').isin(hf_images)
    
    # Add full path to the image
    df_full['img_path'] = df_full['img'].apply(lambda x: hf_data / x)
    
    # Drop rows where the image is not found
    images_not_found = len(df_full[df_full['image_found'] == False])
    logger.info(f"Images not found: {images_not_found}")
    df_full = df_full[df_full['image_found'] == True]
    
    # Drop duplicate images
    num_duplicates = df_full['img'].duplicated().sum()
    logger.info(f"Number of dropped duplicated images: {num_duplicates}")
    df_full = df_full.drop_duplicates(subset=['img'], keep='first')
    
    logger.info(f"Number of labels after cleaning: {len(df_full)}")
    
    return df_full


def load_annotated_subset():
    """Load the annotated multilabel subset."""
    logger.info(f"Loading annotated subset: {cfg.data.final_dataset}")
    
    ds_subset = load_dataset(cfg.data.final_dataset)
    df_subset = pd.concat([
        ds_subset['train'].to_pandas(),
        ds_subset['validation'].to_pandas(),
        ds_subset['test'].to_pandas()
    ])
    
    # Drop image column
    df_subset = df_subset.drop(columns=['image'])
    
    logger.info(f"Loaded {len(df_subset)} annotated samples")
    
    return df_subset


def generate_descriptive_stats(df_full, df_subset):
    """Generate descriptive statistics and return as formatted string."""
    output_lines = []
    
    output_lines.append("="*60)
    output_lines.append("DESCRIPTIVE STATISTICS - HATEFUL MEMES DATASET")
    output_lines.append("="*60)
    output_lines.append("")
    
    # Full dataset statistics
    output_lines.append("FULL DATASET STATISTICS")
    output_lines.append("-"*60)
    output_lines.append(f"Total samples: {len(df_full)}")
    output_lines.append("")
    output_lines.append("Label distribution (normalized):")
    label_dist = df_full["label"].value_counts(normalize=True).round(2)
    for label, proportion in label_dist.items():
        output_lines.append(f"  Label {label}: {proportion:.2f}")
    output_lines.append("")
    
    # Annotated subset statistics
    output_lines.append("="*60)
    output_lines.append("ANNOTATED SUBSET STATISTICS")
    output_lines.append("-"*60)
    output_lines.append(f"Total annotated samples: {len(df_subset)}")
    output_lines.append("")
    
    # Marginal distributions for each label
    for col in ["label_hateful", "label_intolerance", "label_incivility"]:
        output_lines.append(f"Marginal distribution for {col}:")
        col_dist = df_subset[col].value_counts(normalize=True).round(2)
        for label, proportion in col_dist.items():
            output_lines.append(f"  {label}: {proportion:.2f}")
        output_lines.append("")
    
    # Cross-tabulation
    output_lines.append("="*60)
    output_lines.append("CROSS-TABULATION")
    output_lines.append("-"*60)
    output_lines.append("Intolerance vs. Incivility (normalized):")
    output_lines.append("")
    crosstab = pd.crosstab(
        df_subset['label_incivility'], 
        df_subset['label_intolerance'], 
        normalize='all'
    ).round(2)
    output_lines.append(crosstab.to_string())
    output_lines.append("")
    
    output_lines.append("="*60)
    output_lines.append("END OF DESCRIPTIVE STATISTICS")
    output_lines.append("="*60)
    
    return "\n".join(output_lines)


def main():
    """Main execution function."""
    logger.info("Starting descriptive statistics analysis...")
    
    # Load datasets
    df_full = load_full_dataset()
    df_subset = load_annotated_subset()
    
    # Verify all images in subset are in full dataset
    logger.info("Verifying image consistency...")
    missing_images = set(df_subset['img']) - set(df_full['img'])
    if len(missing_images) > 0:
        logger.warning(f"Missing images in full dataset: {missing_images}")
    else:
        logger.info("All subset images found in full dataset ✓")
    
    # Generate statistics
    logger.info("Generating descriptive statistics...")
    stats_text = generate_descriptive_stats(df_full, df_subset)
    
    # Save results
    logger.info("Saving descriptive statistics...")
    output_dir = project_root / cfg.out.results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "descriptive_stats.txt"
    
    with open(output_file, 'w') as f:
        f.write(stats_text)
    
    logger.info(f"Results saved to {output_file}")
    
    # Also print to console
    print("\n" + stats_text)
    
    logger.info("\n" + "="*60)
    logger.info("Descriptive statistics analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
