from pathlib import Path

import json
import pandas as pd

from datasets import load_dataset
from omegaconf import OmegaConf
from scipy.stats import chi2_contingency

from beyond_hate.logger import get_logger

# Resolve project root based on this file's location
project_root = Path(__file__).parent.parent.parent.resolve()

# Load configuration
config_path = project_root / 'config/default.yaml'
cfg = OmegaConf.load(config_path)

# Load logger
logs_dir = project_root / cfg.out.logs
logger = get_logger("bias_analysis", logs_dir=logs_dir)


def analyze_hypothesis(annotations, hypothesis_name, hypothesis_label):
    """
    Analyze a hypothesis by comparing label_hateful with a hypothesis label.
    
    Args:
        annotations: DataFrame with annotation data
        hypothesis_name: Name of the hypothesis for logging
        hypothesis_label: Series or array with hypothesis labels to compare against label_hateful
    
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing {hypothesis_name}")

    total_samples = len(annotations)

    # Calculate statistics
    perfect_match = (annotations["label_hateful"] == hypothesis_label).sum()
    match_rate = perfect_match / total_samples
    corr = annotations["label_hateful"].corr(hypothesis_label)

    logger.info(f"Perfect matches: {perfect_match}/{total_samples} ({match_rate*100:.1f}%)")
    logger.info(f"Correlation: r={corr:.3f}")

    # Chi-square test
    confusion = pd.crosstab(hypothesis_label, annotations["label_hateful"])
    chi2_val, p_val, _, _ = chi2_contingency(confusion)
    chi2_val = float(chi2_val)
    p_val = float(p_val)
    logger.info(f"Chi-square test: χ²={chi2_val:.3f}, p={p_val:.4f}")

    return {
        "perfect_matches": int(perfect_match),
        "total_samples": total_samples,
        "match_rate": round(match_rate, 3),
        "correlation": round(float(corr), 3),
        "chi2": round(chi2_val, 3),
        "p_value": round(p_val, 3)
    }


def main():
    """Main execution function."""
    logger.info("Starting bias analysis...")

    # Load dataset
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    ds = load_dataset(cfg.data.final_dataset)
    logger.info(f"Available splits: {list(ds.keys())}")

    annotations = pd.concat(
        [ds["train"].to_pandas(),
         ds["validation"].to_pandas(),
         ds["test"].to_pandas()],
        ignore_index=True
    )
    logger.info(f"Loaded {len(annotations)} total samples")

    # Analyze all hypotheses
    logger.info("\n" + "="*60)
    logger.info("ANALYZING ALL HYPOTHESES")
    logger.info("="*60)

    h1 = annotations["label_intolerance"]
    h1_results = analyze_hypothesis(
        annotations,
        "HYPOTHESIS 1: Hatefulness = Intolerance",
        h1
    )

    h2 = annotations["label_incivility"]
    h2_results = analyze_hypothesis(
        annotations,
        "HYPOTHESIS 2: Hatefulness = Incivility",
        h2
    )

    h3 = ((annotations["label_incivility"] == 1) | (annotations["label_intolerance"] == 1)).astype(int)
    h3_results = analyze_hypothesis(
        annotations,
        "HYPOTHESIS 3: Hatefulness = (Incivil OR Intolerant)",
        h3
    )
    h4 = ((annotations["label_incivility"] == 1) & (annotations["label_intolerance"] == 1)).astype(int)
    h4_results = analyze_hypothesis(
        annotations,
        "HYPOTHESIS 4: Hatefulness = (Incivil AND Intolerant)",
        h4
    )

    # Compile results
    results = {
        "hypothesis_1_hatefulness_equals_intolerance": h1_results,
        "hypothesis_2_hatefulness_equals_incivil_or_intolerant": h2_results,
        "hypothesis_3_hatefulness_equals_incivility": h3_results,
        "hypothesis_4_hatefulness_equals_incivil_and_intolerant": h4_results,
    }

    # Save results
    logger.info("Saving bias analysis results...")
    output_dir = project_root / cfg.out.results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "bias_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    logger.info("\n" + "="*60)
    logger.info("Bias analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
