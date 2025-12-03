from dotenv import load_dotenv
load_dotenv()

from unsloth import FastVisionModel
from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.auto import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, extract_label, to_inference_conversation
from beyond_hate.train.prompts import coarse_prompt
from beyond_hate.logger import get_logger

def main():
    # Config paths
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    config_base_path = project_root / 'config/default.yaml'
    config_coarse_path = project_root / 'config/coarse.yaml'
    
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
    test_ds = load_dataset(cfg.data.final_dataset, split='test')
    logger.info(f"Loaded {len(test_ds)} test samples")
    
    # Load prompts
    SYSTEM_TEXT = coarse_prompt['system']
    USER_TEXT = coarse_prompt['user']
    
    # Load the fine-tuned model
    checkpoint_path = project_root / cfg.evaluation.checkpoint_path
    logger.info(f"Loading model from: {checkpoint_path}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        str(checkpoint_path),
        load_in_4bit=cfg.training.load_in_4bit,
        use_gradient_checkpointing=cfg.training.use_gradient_checkpointing,
        max_seq_length=cfg.training.max_seq_length
    )
    
    # Set model to inference mode
    FastVisionModel.for_inference(model)
    
    # Initialize WandB for logging evaluation results
    run_name = f"eval_{checkpoint_path.parent.name}"
    logger.info(f"Initializing WandB run: {run_name}")
    wandb.init(
        project=cfg.wandb.project, 
        name=run_name,
        dir=project_root / cfg.out.path,
        config=OmegaConf.to_container(cfg)
    )
    
    # Prepare test dataset
    logger.info("Preparing test dataset...")
    test_dataset_converted = [to_inference_conversation(d, SYSTEM_TEXT, USER_TEXT, 
                                                         img_size=tuple(cfg.training.img_size), 
                                                         img_color_padding=tuple(cfg.training.img_color_padding))
                              for d in tqdm(test_ds)]
    
    # Run inference on test data
    logger.info("Running inference on test data...")
    results = []
    for conversation, image, data_id, labels in tqdm(test_dataset_converted):
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = tokenizer(images=image, text=prompt, return_tensors="pt").to("cuda:0")
        
        # Generate prediction
        max_new_tokens = 50
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.split('[/INST]')[-1].strip()
        
        results.append({
            'id': data_id,
            'label': labels['label_hateful'],
            'output': output,
        })
    
    # Calculate metrics
    possible_labels = {'Hateful': 1, 'Neutral': 0}
    
    y_true = [r['label'] for r in results]
    y_pred = [extract_label(r['output'], possible_labels) for r in results]
    
    # Evaluate
    evaluation = binary_evaluation(y_true, y_pred)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall: {evaluation['recall']:.4f}")
    logger.info(f"  F1 Score: {evaluation['f1_score']:.4f}")
    logger.info(f"  Invalid Prediction Rate: {evaluation['invalid_prediction_rate']:.4f}")
        
    # Log confusion matrix to wandb
    valid = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true_valid = [y_true[i] for i in valid]
    y_pred_valid = [y_pred[i] for i in valid]
    
    logger.info(f"Valid predictions: {len(y_true_valid)}/{len(y_true)} ({len(y_true_valid)/len(y_true)*100:.1f}%)")

    wandb.log({"eval/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true_valid,
        preds=y_pred_valid,
        class_names=['Neutral', 'Hateful']
    )})
    
    # Log metrics to wandb
    wandb.log({
        "eval/accuracy": evaluation['accuracy'],
        "eval/precision": evaluation['precision'],
        "eval/recall": evaluation['recall'],
        "eval/f1_score": evaluation['f1_score'],
        "eval/total_samples": len(y_true),
        "eval/valid_samples": len(y_true_valid),
        "eval/invalid_prediction_rate": evaluation['invalid_prediction_rate']
    })
    
    # Finish wandb run
    logger.info("Evaluation complete!")
    wandb.finish()

if __name__ == "__main__":
    main()