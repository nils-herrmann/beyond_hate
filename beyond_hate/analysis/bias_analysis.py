from pathlib import Path

import json
import pandas as pd

from omegaconf import OmegaConf

from beyond_hate.logger import get_logger

# Resolve project root based on this file's location
project_root = Path(__file__).parent.parent.parent.resolve()

# Load configuration
config_path = project_root / 'config/default.yaml'
cfg = OmegaConf.load(config_path)

# Load logger
logs_dir = project_root / cfg.out.logs
logger = get_logger("bias_analysis", logs_dir=logs_dir)


def analyze_predictions(df, label_col, pred_col):
    """
    Analyze binary classification results for any label/prediction pair.
    
    Args:
        df: DataFrame with results
        label_col: Name of the label column (e.g., 'label_hateful')
        pred_col: Name of the prediction column (e.g., 'pred_hateful')
    
    Returns:
        Dictionary with analysis results including accuracy metrics and confusion matrix
    """
    # Extract the category name from the prediction column
    category = pred_col.replace('pred_', '')
    
    # Overall accuracy
    overall_accuracy = (df[label_col] == df[pred_col]).mean()

    # Class-specific accuracy
    positive_mask = df[label_col] == 1
    negative_mask = df[label_col] == 0
    
    positive_accuracy = (df[positive_mask][label_col] == df[positive_mask][pred_col]).mean()
    negative_accuracy = (df[negative_mask][label_col] == df[negative_mask][pred_col]).mean()

    # Confusion matrix components
    false_positives = ((df[label_col] == 0) & (df[pred_col] == 1)).sum()
    false_negatives = ((df[label_col] == 1) & (df[pred_col] == 0)).sum()
    true_negatives = ((df[label_col] == 0) & (df[pred_col] == 0)).sum()
    true_positives = ((df[label_col] == 1) & (df[pred_col] == 1)).sum()
    
    # Error rates
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
    
    return {
        "category": category,
        "overall_accuracy": round(float(overall_accuracy), 2),
        f"{category}_accuracy": round(float(positive_accuracy), 2),
        f"non_{category}_accuracy": round(float(negative_accuracy), 2),
        "false_positive_rate": round(float(fpr), 2),
        "false_negative_rate": round(float(fnr), 2),
        "error_difference": round(float(fnr - fpr), 2),
        "counts": {
            "true_positives": int(true_positives),
            "true_negatives": int(true_negatives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives)
        }
    }


def detect_prediction_columns(df):
    """
    Detect which prediction columns are present in the DataFrame.
    
    Args:
        df: DataFrame with results
    
    Returns:
        List of tuples (label_col, pred_col) for each detected prediction
    """
    prediction_pairs = []
    
    # Known prediction types
    known_types = ['hateful', 'intolerance', 'incivility']
    
    for pred_type in known_types:
        label_col = f'label_{pred_type}'
        pred_col = f'pred_{pred_type}'
        
        if label_col in df.columns and pred_col in df.columns:
            prediction_pairs.append((label_col, pred_col))
            logger.info(f"  Found prediction pair: {label_col} / {pred_col}")
    
    return prediction_pairs


def analyze_results_for_task(task_name, result_configs):
    """
    Analyze results for a specific task (e.g., coarse, fine, joint).
    
    Args:
        task_name: Name of the task (e.g., 'coarse', 'fine', 'joint')
        result_configs: List of result configurations from config
    
    Returns:
        Dictionary with analysis results for all models/runs
    """
    logger.info("\n" + "="*60)
    logger.info(f"ANALYZING {task_name.upper()} RESULTS")
    logger.info("="*60)
    
    results_path = project_root / cfg.out.results / task_name
    task_results = {}
    
    for result_config in result_configs:
        model = result_config['model']
        run_id = result_config['run_id']
        
        logger.info(f"\nModel: {model}")
        logger.info(f"Run ID: {run_id}")
        
        results_file = results_path / f"{run_id}_results.jsonl"
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            continue
        
        # Load results
        df = pd.read_json(results_file, lines=True)
        logger.info(f"Loaded {len(df)} samples")
        
        # Detect available prediction columns
        logger.info("Detecting prediction columns...")
        prediction_pairs = detect_prediction_columns(df)
        
        if not prediction_pairs:
            logger.warning(f"No prediction columns found in {results_file}")
            continue
        
        # Analyze each prediction type
        analyses = {}
        for label_col, pred_col in prediction_pairs:
            category = pred_col.replace('pred_', '')
            logger.info(f"\nAnalyzing {category}...")
            
            analysis = analyze_predictions(df, label_col, pred_col)
            
            # Log results
            logger.info(f"  Overall Accuracy: {analysis['overall_accuracy']:.2f}")
            logger.info(f"  {category.capitalize()} Accuracy: {analysis[f'{category}_accuracy']:.2f}")
            logger.info(f"  Non-{category.capitalize()} Accuracy: {analysis[f'non_{category}_accuracy']:.2f}")
            logger.info(f"  False Positive Rate (FPR): {analysis['false_positive_rate']:.2f}")
            logger.info(f"  False Negative Rate (FNR): {analysis['false_negative_rate']:.2f}")
            logger.info(f"  Error Difference (FNR - FPR): {analysis['error_difference']:.2f}")
            
            analyses[category] = analysis
        
        # Store results
        task_results[run_id] = {
            "model": model,
            "run_id": run_id,
            "analyses": analyses
        }
    
    return task_results


def main():
    """Main execution function."""
    logger.info("Starting bias analysis...")

    results = {}

    # Analyze all task types defined in config
    for task_name in ['coarse', 'fine', 'joint']:
        if hasattr(cfg.results_ids, task_name):
            result_configs = getattr(cfg.results_ids, task_name)
            if result_configs:
                results[task_name] = analyze_results_for_task(task_name, result_configs)

    # Save results
    logger.info("\n" + "="*60)
    logger.info("Saving bias analysis results...")
    output_dir = project_root / cfg.out.results / "analysis"
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
