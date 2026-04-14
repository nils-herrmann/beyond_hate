from dotenv import load_dotenv
load_dotenv()

import json
import time
import torch
import wandb
from unsloth import FastVisionModel
from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.auto import tqdm

from beyond_hate.train.utils import binary_evaluation, extract_label, to_inference_conversation
from beyond_hate.train.prompts import coarse_prompt
from beyond_hate.logger import get_logger


def evaluate_model(cfg, model_id, task_name, logger, project_root, test_ds, SYSTEM_TEXT, USER_TEXT):
    """Evaluate a single model/checkpoint."""
    
    # Determine if it's a checkpoint or base model
    if model_id.startswith("unsloth/"):
        run_name = f"eval_{task_name}_{model_id.replace('/', '_')}"
        logger.info(f"Loading pretrained model: {model_id}")
    else:
        checkpoint_path = project_root / model_id
        run_name = f"eval_{task_name}_{checkpoint_path.parent.name}"
        model_id = str(checkpoint_path)
        logger.info(f"Loading fine-tuned model from: {model_id}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=cfg.training.load_in_4bit,
        use_gradient_checkpointing=cfg.training.use_gradient_checkpointing,
        max_seq_length=cfg.training.max_seq_length
    )
    
    # Set model to inference mode
    FastVisionModel.for_inference(model)
    
    # Initialize WandB for logging evaluation results
    logger.info(f"Initializing WandB run: {run_name}")
    wandb.init(
        project=cfg.wandb.project, 
        name=run_name,
        dir=project_root / cfg.out.path,
        config=OmegaConf.to_container(cfg)
    )
    run_id = wandb.run.id
    
    # Prepare test dataset
    logger.info("Preparing test dataset...")
    test_dataset_converted = [to_inference_conversation(d, SYSTEM_TEXT, USER_TEXT, 
                                                         img_size=tuple(cfg.training.img_size), 
                                                         img_color_padding=tuple(cfg.training.img_color_padding))
                              for d in tqdm(test_ds)]
    
    # Run inference on test data
    logger.info("Running inference on test data...")
    batch_size = cfg.evaluation.batch_size
    results = []
    # Process in batches
    for i in tqdm(range(0, len(test_dataset_converted), batch_size), desc="Evaluating batches"):
        batch = test_dataset_converted[i:i+batch_size]
        
        # Prepare batch data
        conversations = [item[0] for item in batch]
        images = [item[1] for item in batch]
        data_ids = [item[2] for item in batch]
        labels_list = [item[3] for item in batch]
        
        # Apply chat template and tokenize batch
        prompts = [tokenizer.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        inputs = tokenizer(images=images, text=prompts, return_tensors="pt", padding=True).to("cuda:0")
        
        # Generate for batch
        max_new_tokens = 50
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Extract only generated tokens (model-agnostic approach)
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Process each output in the batch
        possible_labels = {"Hateful": 1, "Neutral": 0}
        for data_id, labels, decoded_output in zip(data_ids, labels_list, decoded_outputs):
            output = decoded_output.strip()
            pred = extract_label(output, possible_labels)
            results.append({
                "id": data_id,
                "label_hateful": labels["label_hateful"],
                "label_incivility": labels["label_incivility"],
                "label_intolerance": labels["label_intolerance"],
                "output": output,
                "pred_hateful": pred
            })
    
    # Extract true labels and predictions
    y_true = [r["label_hateful"] for r in results]
    y_pred = [r["pred_hateful"] for r in results]
    
    # Evaluate
    evaluation = binary_evaluation(y_true, y_pred)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall: {evaluation['recall']:.4f}")
    logger.info(f"  F1 Score: {evaluation['f1_score']:.4f}")
    logger.info(f"  Invalid Prediction Rate: {evaluation['invalid_prediction_rate']:.4f}")
    
    # Log metadata
    wandb.log({"model": model.config._name_or_path})
    # Log metrics to wandb
    wandb.log({
        "test/hateful/accuracy": evaluation["accuracy"],
        "test/hateful/precision": evaluation["precision"],
        "test/hateful/recall": evaluation["recall"],
        "test/hateful/f1_score": evaluation["f1_score"],
        "test/hateful/confusion_matrix": evaluation['confusion_matrix'].tolist(),
        "test/total_samples": len(y_true),
        "test/valid_samples": len(y_true_valid),
        "test/invalid_prediction_rate": evaluation["invalid_prediction_rate"]
    })
    
    # Finish wandb run
    logger.info("Evaluation complete!")
    wandb.finish()
    
    # Save results to a file
    out_path = Path(cfg.out.results) / "coarse"
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%y%m%d-%H%M%S")
    results_path = out_path / f"{run_id}_results.jsonl"
    logger.info(f"Saving detailed results to: {results_path}")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Save evaluation metrics
    metrics_path = out_path / f"{run_id}_metrics.json"
    metrics = {
        'model': model.config._name_or_path,
        'task': task_name,
        'total_samples': len(y_true),
        'valid_samples': len(y_true_valid),
        'accuracy': float(evaluation['accuracy']),
        'precision': float(evaluation['precision']),
        'recall': float(evaluation['recall']),
        'f1_score': float(evaluation['f1_score']),
        'invalid_prediction_rate': float(evaluation['invalid_prediction_rate']),
        'confusion_matrix': evaluation['confusion_matrix'].tolist()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    return metrics


def main():
    # Config paths
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    config_base_path = project_root / "config/default.yaml"
    config_coarse_path = project_root / "config/coarse.yaml"
    
    # Load configurations
    cfg = OmegaConf.load(config_base_path)
    custom_cfg = OmegaConf.load(config_coarse_path)
    
    # Override default config with custom config
    cfg = OmegaConf.merge(cfg, custom_cfg)
    
    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("eval_coarse", logs_dir=logs_dir)
    
    logger.info("Starting coarse-grained evaluation...")
    
    # Load the test data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    test_ds = load_dataset(cfg.data.final_dataset, split="test")
    logger.info(f"Loaded {len(test_ds)} test samples")
    
    # Load prompts
    SYSTEM_TEXT = coarse_prompt["system"]
    USER_TEXT = coarse_prompt["user"]
    
    # Iterate over tasks
    tasks = OmegaConf.select(cfg, "evaluation.tasks")
    if tasks is None:
        logger.error("No evaluation tasks found in config")
        return
    
    for task_name, task_list in tasks.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing task: {task_name}")
        logger.info(f"{'='*80}")
        
        for task_config in task_list:
            # Get model or checkpoint path
            model_id = task_config.get("model") or task_config.get("checkpoint_path")
            if not model_id:
                logger.warning(f"Skipping task config without model or checkpoint_path: {task_config}")
                continue
            
            logger.info(f"\nEvaluating: {model_id}")
            
            try:
                metrics = evaluate_model(cfg, model_id, task_name, logger, project_root, 
                                       test_ds, SYSTEM_TEXT, USER_TEXT)
                logger.info(f"Completed evaluation for {model_id}")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_id}: {str(e)}")
                continue
    
    logger.info("\nAll evaluations complete!")

if __name__ == "__main__":
    main()