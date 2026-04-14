from pathlib import Path

import json
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
from datasets import Image as HFImage
from dotenv import dotenv_values
from omegaconf import OmegaConf
from tqdm.auto import trange

from beyond_hate.logger import get_logger
from beyond_hate.analysis.utils import compute_pairwise_agreement, label_summary, split_dataset

# Resolve project root based on this file's location
project_root = Path(__file__).parent.parent.parent.resolve()

# Load configuration
config_path = project_root / 'config/default.yaml'
cfg = OmegaConf.load(config_path)

# Load logger
logs_dir = project_root / cfg.out.logs
logger = get_logger("validate_annotations", logs_dir=logs_dir)

def validate_annotations(annotations, relevant_ids):
    """Validate annotation data."""
    logger.info("Validating annotations...")

    # Check if all images were annotated by each annotator
    annotator_counts = annotations.groupby("annotator")["id"].nunique()
    expected_count = len(relevant_ids)
    for annotator_id, count in annotator_counts.items():
        if count != expected_count:
            logger.warning(f"Annotator {annotator_id} annotated {count} images, expected {expected_count}")

    # Check if 0 label and other labels are mutually exclusive
    def has_zero_with_others(label_str):
        labels = label_str.split(',')
        return '0' in labels and len(labels) > 1

    incivility_violations = annotations['label_incivility'].apply(has_zero_with_others).sum()
    intolerance_violations = annotations['label_intolerance'].apply(has_zero_with_others).sum()

    if incivility_violations > 0:
        logger.warning(f"Found {incivility_violations} incivility labels with '0' mixed with others")
    if intolerance_violations > 0:
        logger.warning(f"Found {intolerance_violations} intolerance labels with '0' mixed with others")

    # Drop duplicates by annotator (keep last)
    annotations = annotations.drop_duplicates(subset=["annotator", "id"], keep="last")

    # Binarize labels for agreement calculation
    annotations = annotations.copy()
    annotations["label_incivility_bin"]  = annotations["label_incivility"].apply(lambda x: 1 if "0" not in x else 0)
    annotations["label_intolerance_bin"] = annotations["label_intolerance"].apply(lambda x: 1 if "0" not in x else 0)

    logger.info("Annotation validation complete")
    return annotations

def compute_agreement(annotations):
    """Compute inter-annotator agreement."""
    logger.info("Computing inter-annotator agreement...")

    annotators = annotations["annotator"].unique()
    logger.info(f"Number of annotators: {len(annotators)}")
    logger.info(f"Annotators: {list(annotators)}")

    agreement_results = {}

    # Hateful labels
    logger.info("Computing agreement for HATEFUL LABELS...")
    hate_agreements, hate_kappas = compute_pairwise_agreement(annotations, "label_hateful")
    if hate_agreements:
        hate_result = {
            "average_agreement": round(float(np.mean(hate_agreements)), 3),
            "std_agreement": round(float(np.std(hate_agreements)), 3),
            "average_kappa": round(float(np.mean(hate_kappas)), 3),
            "std_kappa": round(float(np.std(hate_kappas)), 3)
        }
        agreement_results["label_hateful"] = hate_result
        logger.info(f"Average agreement: {hate_result['average_agreement']:.3f} ± {hate_result['std_agreement']:.3f}")
        logger.info(f"Average Kappa: {hate_result['average_kappa']:.3f} ± {hate_result['std_kappa']:.3f}")

    # Incivility labels
    logger.info("Computing agreement for INCIVILITY LABELS...")
    inciv_agreements, inciv_kappas = compute_pairwise_agreement(annotations, "label_incivility_bin")
    if inciv_agreements:
        inciv_result = {
            "average_agreement": round(float(np.mean(inciv_agreements)), 3),
            "std_agreement": round(float(np.std(inciv_agreements)), 3),
            "average_kappa": round(float(np.mean(inciv_kappas)), 3),
            "std_kappa": round(float(np.std(inciv_kappas)), 3)
        }
        agreement_results["label_incivility"] = inciv_result
        logger.info(f"Average agreement: {inciv_result['average_agreement']:.3f} ± {inciv_result['std_agreement']:.3f}")
        logger.info(f"Average Kappa: {inciv_result['average_kappa']:.3f} ± {inciv_result['std_kappa']:.3f}")

    # Intolerance labels
    logger.info("Computing agreement for INTOLERANCE LABELS...")
    intol_agreements, intol_kappas = compute_pairwise_agreement(annotations, "label_intolerance_bin")
    if intol_agreements:
        intol_result = {
            "average_agreement": round(float(np.mean(intol_agreements)), 3),
            "std_agreement": round(float(np.std(intol_agreements)), 3),
            "average_kappa": round(float(np.mean(intol_kappas)), 3),
            "std_kappa": round(float(np.std(intol_kappas)), 3)
        }
        agreement_results["label_intolerance"] = intol_result
        logger.info(f"Average agreement: {intol_result['average_agreement']:.3f} ± {intol_result['std_agreement']:.3f}")
        logger.info(f"Average Kappa: {intol_result['average_kappa']:.3f} ± {intol_result['std_kappa']:.3f}")

    return agreement_results

def aggregate_annotations(annotations):
    """Aggregate annotations using majority voting."""
    logger.info("Aggregating annotations via majority voting...")

    annotations_agg = annotations.groupby(["id", "text"]).agg({
        "label_hateful": "mean",
        "label_incivility_bin": "mean",
        "label_intolerance_bin": "mean"
    }).reset_index()

    annotations_agg["label_hateful"] = annotations_agg["label_hateful"].apply(lambda x: 1 if x >= 0.5 else 0)
    annotations_agg["label_incivility"] = annotations_agg["label_incivility_bin"].apply(lambda x: 1 if x >= 0.5 else 0)
    annotations_agg["label_intolerance"] = annotations_agg["label_intolerance_bin"].apply(lambda x: 1 if x >= 0.5 else 0)

    logger.info(f"Aggregated {len(annotations_agg)} annotations")
    return annotations_agg

def map_splits_and_images(annotations_agg, hf_data_path):
    """Map splits and image paths to annotations."""
    logger.info("Mapping splits and image paths...")

    original_df = pd.DataFrame()
    splits = ['dev_seen', 'dev_unseen', 'test_seen', 'test_unseen', 'train']
    for split in splits:
        df_tmp = pd.read_json(f'{hf_data_path}/{split}.jsonl', lines=True)
        df_tmp['split'] = split
        original_df = pd.concat([original_df, df_tmp])

    # Drop duplicates by id before merging to avoid inflating annotations_agg
    original_df = original_df.drop_duplicates(subset=['id'], keep='first')

    annotations_agg = annotations_agg.merge(original_df[['id', 'img', 'split']], on='id', how='left')

    # Check for missing splits or imgs
    missing_splits = annotations_agg['split'].isnull().sum()
    missing_img = annotations_agg['img'].isnull().sum()
    if (missing_splits > 0) or (missing_img > 0):
        logger.warning(f"There are {missing_splits} annotations with missing splits and {missing_img} with missing images")

    # Keep only relevant columns
    annotations_agg = annotations_agg[["id", "text" ,"img", "split", "label_hateful", "label_incivility", "label_intolerance"]]

    logger.info(f"Mapped splits for {len(annotations_agg)} annotations")
    return annotations_agg

def find_best_seed(annotations_agg, label_cols, n_seeds=1000):
    """Find the best seed that minimizes label distribution drift."""
    logger.info(f"Searching for best seed among {n_seeds} candidates...")

    drift_results = []

    for seed in trange(n_seeds, desc="Testing seeds"):
        train_df, val_df, test_df = split_dataset(
            annotations_agg,
            train_share=0.7, val_share=0.1, test_share=0.2,
            stratfy=["split"],
            seed=seed
        )
        _, drift = label_summary(train_df, val_df, test_df, label_cols)
        drift_results.append((seed, drift))

    best_seed, best_drift = min(drift_results, key=lambda x: x[1])

    logger.info(f"Best seed: {best_seed} with total drift = {best_drift:.6f}")

    return best_seed

def split_and_summarize(annotations_agg, label_cols, best_seed):
    """Split dataset and summarize label balance."""
    logger.info(f"Splitting dataset with seed {best_seed}...")

    train_df, val_df, test_df = split_dataset(
        annotations_agg,
        train_share=0.7, val_share=0.1, test_share=0.2,
        seed=best_seed
    )

    final_summary, _ = label_summary(train_df, val_df, test_df, label_cols)

    logger.info("Final label balance summary:")
    logger.info(final_summary.to_string())

    return train_df, val_df, test_df, final_summary



def main():
    """Main execution function."""
    logger.info("Starting annotation validation...")

    # Load environment variables
    env_values = dotenv_values()
    hf_token = env_values.get("HF_TOKEN")

    # Build paths
    labels_file = project_root / cfg.data.paths.labels_file
    images_to_annotate_file = project_root / cfg.data.paths.base / "images_to_annotate.txt"
    hf_data_path = project_root / cfg.data.paths.hf
    output_dir = project_root / cfg.out.results / "analysis"

    # Load and validate annotations
    logger.info("Loading relevant ids and annotations...")

    # Load relevant ids
    with open(images_to_annotate_file, "r") as f:
        lines = f.readlines()
        relevant_ids = [int(line.strip()) for line in lines]

    logger.info(f"Loaded {len(relevant_ids)} relevant image IDs")

    # Load annotations and keep only relevant ones
    annotations = pd.read_json(labels_file, lines=True,
                               dtype={"label_hateful": int,
                                      "label_incivility": str,
                                      "label_intolerance": str})
    annotations = annotations[annotations["id"].isin(relevant_ids)]

    logger.info(f"Loaded {len(annotations)} annotations")
    
    annotations = validate_annotations(annotations, relevant_ids)

    # Compute inter-annotator agreement
    agreement_results = compute_agreement(annotations)

    # Aggregate annotations
    annotations_agg = aggregate_annotations(annotations)

    # Map splits and images
    annotations_agg = map_splits_and_images(annotations_agg, hf_data_path)

    # Find best seed and split dataset
    label_cols = ["label_hateful", "label_incivility", "label_intolerance"]
    best_seed = find_best_seed(annotations_agg, label_cols, n_seeds=1000)
    train_df, val_df, test_df, final_summary = split_and_summarize(annotations_agg, label_cols, best_seed)

    # Save results
    logger.info(f"Saving results to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save agreement results
    agreement_file = output_dir / "inter_annotator_agreement.json"
    with open(agreement_file, 'w') as f:
        json.dump(agreement_results, f, indent=2)
    logger.info(f"Agreement results saved to {agreement_file}")

    # Save label balance summary
    summary_file = output_dir / "label_balance_summary.json"
    summary_dict = final_summary.to_dict(orient='index')
    with open(summary_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Label balance summary saved to {summary_file}")

    # Create HuggingFace datasets with images
    logger.info("Adding images to datasets...")

    # Add image paths to dataframes
    def add_images_to_df(df, hf_data_path):
        """Add image paths to dataframe for HuggingFace encoding."""
        image_paths = []
        for img_path in df['img']:
            full_path = hf_data_path / img_path
            image_paths.append(str(full_path))
        df = df.copy()
        df['image'] = image_paths
        return df

    train_df_with_images = add_images_to_df(train_df, hf_data_path)
    val_df_with_images = add_images_to_df(val_df, hf_data_path)
    test_df_with_images = add_images_to_df(test_df, hf_data_path)

    # Create HuggingFace datasets
    logger.info("Creating HuggingFace datasets...")
    hf_train = Dataset.from_pandas(train_df_with_images.reset_index(drop=True))
    hf_val = Dataset.from_pandas(val_df_with_images.reset_index(drop=True))
    hf_test = Dataset.from_pandas(test_df_with_images.reset_index(drop=True))

    # Cast the 'image' column to HuggingFace Image type
    hf_train = hf_train.cast_column('image', HFImage())
    hf_val = hf_val.cast_column('image', HFImage())
    hf_test = hf_test.cast_column('image', HFImage())

    hf_ds = DatasetDict({
        "train": hf_train,
        "validation": hf_val,
        "test": hf_test,
    })

    # Upload to Hugging Face Hub
    logger.info(f"Uploading dataset to Hugging Face Hub ({cfg.data.final_dataset})...")
    hf_ds.push_to_hub(
        cfg.data.final_dataset,
        token=hf_token,
        private=False
    )
    logger.info(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{cfg.data.final_dataset}")

    logger.info("Annotation validation complete!")

if __name__ == "__main__":
    main()
