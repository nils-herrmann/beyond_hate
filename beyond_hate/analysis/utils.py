import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score
from warnings import warn

def compute_pairwise_agreement(df, label_col):
    """
    Compute pairwise agreement and Cohen's Kappa between annotators for a given label column.
    Args:
        df: pandas DataFrame with columns 'annotator', 'id', and the specified label_col
        label_col: string, name of the column containing the labels to compare
    Returns:
        agreements: list of agreement scores between annotator pairs
        kappas: list of Cohen's Kappa scores between annotator pairs
    """
    annotators = df["annotator"].unique()
    agreements = []
    kappas = []
    
    for i, ann1 in enumerate(annotators):
        for ann2 in annotators[i+1:]:
            # Get common images annotated by both
            ann1_data = df[df["annotator"] == ann1].set_index("id")[label_col]
            ann2_data = df[df["annotator"] == ann2].set_index("id")[label_col]
            
            # Find intersection
            common_ids = ann1_data.index.intersection(ann2_data.index)
            
            if len(common_ids) > 0:
                ann1_labels = ann1_data.loc[common_ids].values
                ann2_labels = ann2_data.loc[common_ids].values
                
                # Simple agreement
                agreement = (ann1_labels == ann2_labels).mean()
                agreements.append(agreement)
                
                # Cohen's Kappa
                kappa = cohen_kappa_score(ann1_labels, ann2_labels)
                kappas.append(kappa)
                
                print(f"{ann1} vs {ann2}: {len(common_ids)} common images, Agreement: {agreement:.3f}, Kappa: {kappa:.3f}")

                if len(common_ids) / len(ann1_data) != 1:
                    warn(f"Annotators {ann1} and {ann2} do not have full overlap in annotations.")

    
    return agreements, kappas


def split_dataset(df, train_share=0.7, val_share=0.1, test_share=0.2, stratfy=["split"] ,seed=42):
    """
    Perform a 2-step stratified split using `df['split']`.
    Returns: train_df, val_df, test_df
    """
    assert abs(train_share + val_share + test_share - 1.0) < 1e-6, \
        "Shares must sum to 1."

    # first split into train + temp
    temp_share = val_share + test_share
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_share,
        stratify=df[stratfy],
        random_state=seed,
    )

    # split temp into val + test (relative)
    val_rel = val_share / (val_share + test_share)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_rel,
        stratify=temp_df[stratfy],
        random_state=seed,
    )

    return train_df, val_df, test_df


def label_summary(train_df, val_df, test_df, label_cols) -> tuple[pd.DataFrame, float]:
    """
    Computes positive rates and drift for each label.
    Returns: summary DataFrame + total drift scalar.
    """
    summary = pd.concat(
        {
            "train": train_df[label_cols].mean().rename("positive_rate"),
            "val":   val_df[label_cols].mean().rename("positive_rate"),
            "test":  test_df[label_cols].mean().rename("positive_rate"),
        },
        axis=1,
    )

    # drift for each label
    summary["avg_drift_from_train"] = (
        (summary["val"] - summary["train"]).abs()
        + (summary["test"] - summary["train"]).abs()
    ) / 2

    # scalar metric to optimize
    total_drift = summary["avg_drift_from_train"].mean()

    return summary, total_drift


# -------------------------------------------------------------------------------------------------
from collections import Counter

def parse_labels(label_series):
    """Parse comma-separated integers and flatten into a list"""
    all_labels = []
    for entry in label_series.dropna():
        # Handle string entries with comma-separated values
        if isinstance(entry, str):
            labels = [int(x.strip()) for x in entry.split(',') if x.strip()]
            all_labels.extend(labels)
    return all_labels

def majority_vote_binary(series):
    """Return majority vote for binary annotations."""
    counts = series.value_counts()
    if counts.empty:
        return np.nan
    return counts.idxmax()

def sanitize(name):
    """Make category names safe as column names."""
    return name.replace(" ", "_").replace("/", "_")

def majority_vote_multilabel(series, categories):
    """
    Majority vote for comma-separated multilabel annotations.
    Returns dict of label -> 0/1 indicating majority presence using category names.
    """
    counter = Counter()

    for labels in series.dropna():
        for l in str(labels).split(","):
            counter[int(l)] += 1

    result = {sanitize(categories[i]): 0 for i in categories.keys()}

    for label, count in counter.items():
        if count >= 2:  # majority (>=2 out of 3)
            result[sanitize(categories[label])] = 1

    return pd.Series(result)
