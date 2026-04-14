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

from beyond_hate.train.utils import binary_evaluation, extract_joint_labels, to_inference_conversation
from beyond_hate.train.prompts import joint_prompt
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
        max_new_tokens = 75  # Increased for joint output format
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Extract only generated tokens (model-agnostic approach)
        generated_tokens = outputs[:, input_length:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Process each output in the batch
        for data_id, labels, decoded_output in zip(data_ids, labels_list, decoded_outputs):
            output = decoded_output.strip()
            pred_incivility, pred_intolerance, pred_hateful = extract_joint_labels(output)
            results.append({
                "id": data_id,
                "label_intolerance": labels["label_intolerance"],
                "label_incivility": labels["label_incivility"],
                "label_hateful": labels["label_hateful"],
                "output": output,
                "pred_incivility": pred_incivility,
                "pred_intolerance": pred_intolerance,
                "pred_hateful": pred_hateful
            })
    
    # Extract true labels and predictions
    y_true_incivility = [r["label_incivility"] for r in results]
    y_true_intolerance = [r["label_intolerance"] for r in results]
    y_true_hateful = [r["label_hateful"] for r in results]
    
    y_pred_incivility = [r["pred_incivility"] for r in results]
    y_pred_intolerance = [r["pred_intolerance"] for r in results]
    y_pred_hateful = [r["pred_hateful"] for r in results]
    
    # Evaluate the predictions
    evaluation = {}
    evaluation["incivility"] = binary_evaluation(y_true_incivility, y_pred_incivility)
    evaluation["intolerance"] = binary_evaluation(y_true_intolerance, y_pred_intolerance)
    evaluation["hateful"] = binary_evaluation(y_true_hateful, y_pred_hateful)
    
    # Calculate average metrics
    avg_accuracy = (evaluation["incivility"]["accuracy"] + evaluation["intolerance"]["accuracy"] + evaluation["hateful"]["accuracy"]) / 3
    avg_f1 = (evaluation["incivility"]["f1_score"] + evaluation["intolerance"]["f1_score"] + evaluation["hateful"]["f1_score"]) / 3
    avg_invalid_prediction_rate = (evaluation["incivility"]["invalid_prediction_rate"] + evaluation["intolerance"]["invalid_prediction_rate"] + evaluation["hateful"]["invalid_prediction_rate"]) / 3
    
    logger.info("Evaluation Results:")
    logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"  Average F1: {avg_f1:.4f}")
    logger.info(f"  Incivility - Accuracy: {evaluation['incivility']['accuracy']:.4f}, F1: {evaluation['incivility']['f1_score']:.4f}")
    logger.info(f"  Intolerance - Accuracy: {evaluation['intolerance']['accuracy']:.4f}, F1: {evaluation['intolerance']['f1_score']:.4f}")
    logger.info(f"  Hateful - Accuracy: {evaluation['hateful']['accuracy']:.4f}, F1: {evaluation['hateful']['f1_score']:.4f}")
    
    # Log metadata
    wandb.log({"model": model.config._name_or_path})
    # Log metrics to wandb
    wandb.log({
        "test/accuracy": avg_accuracy,
        "test/f1": avg_f1,
        "test/invalid_prediction_rate": avg_invalid_prediction_rate,
        "test/total_samples": len(y_true_incivility),
    })
    
    for label in ["incivility", "intolerance", "hateful"]:
        wandb.log({
            f"test/{label}/invalid_prediction_rate": evaluation[label]["invalid_prediction_rate"],
            f"test/{label}/accuracy": evaluation[label]["accuracy"],
            f"test/{label}/precision": evaluation[label]["precision"],
            f"test/{label}/recall": evaluation[label]["recall"],
            f"test/{label}/f1": evaluation[label]["f1_score"],
            f"test/{label}/confusion_matrix": evaluation[label]['confusion_matrix'].tolist(),
        })
    
    # Finish wandb run
    logger.info("Evaluation complete!")
    wandb.finish()

    # Save results to a file
    out_path = Path(cfg.out.results) / "joint"
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
        'total_samples': len(y_true_incivility),
        'avg_accuracy': float(avg_accuracy),
        'avg_f1': float(avg_f1),
        'avg_invalid_prediction_rate': float(avg_invalid_prediction_rate),
    }
    
    for label in ["incivility", "intolerance", "hateful"]:
        metrics[label] = {
            'accuracy': float(evaluation[label]['accuracy']),
            'precision': float(evaluation[label]['precision']),
            'recall': float(evaluation[label]['recall']),
            'f1_score': float(evaluation[label]['f1_score']),
            'invalid_prediction_rate': float(evaluation[label]['invalid_prediction_rate']),
            'confusion_matrix': evaluation[label]['confusion_matrix'].tolist()
        }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    return metrics


def main():
    # Config paths
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    config_base_path = project_root / "config/default.yaml"
    config_joint_path = project_root / "config/joint.yaml"
    
    # Load configurations
    cfg = OmegaConf.load(config_base_path)
    custom_cfg = OmegaConf.load(config_joint_path)
    
    # Override default config with custom config
    cfg = OmegaConf.merge(cfg, custom_cfg)
    
    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("eval_joint", logs_dir=logs_dir)
    
    logger.info("Starting joint learning evaluation...")
    
    # Define system and user text from prompts
    SYSTEM_TEXT = joint_prompt["system"]
    USER_TEXT = joint_prompt["user"]
    
    # Load the test data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    test_ds = load_dataset(cfg.data.final_dataset, split="test")
    logger.info(f"Loaded {len(test_ds)} test samples")
    
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
                logger.info(f"  Avg Accuracy: {metrics['avg_accuracy']:.4f}, Avg F1: {metrics['avg_f1']:.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_id}: {str(e)}")
                continue
    
    logger.info("\nAll evaluations complete!")

if __name__ == "__main__":
    main()
