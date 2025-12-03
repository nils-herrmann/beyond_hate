from dotenv import load_dotenv
load_dotenv()

from unsloth import FastVisionModel
from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.auto import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, extract_multi_labels, to_inference_conversation
from beyond_hate.train.prompts import fine_prompt

def main():
    # Config paths
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    config_base_path = project_root / 'config/default.yaml'
    config_fine_path = project_root / 'config/fine.yaml'
    
    # Load configurations
    cfg = OmegaConf.load(config_base_path)
    custom_cfg = OmegaConf.load(config_fine_path)
    
    # Override default config with custom config
    cfg = OmegaConf.merge(cfg, custom_cfg)
    
    # Define system and user text from prompts
    SYSTEM_TEXT = fine_prompt['system']
    USER_TEXT = fine_prompt['user']
    
    # Load the test data
    test_ds = load_dataset(cfg.data.final_dataset, split='test')
    
    print(f"Loaded {len(test_ds)} test samples")
    
    # Load the fine-tuned model
    checkpoint_path = project_root / cfg.evaluation.checkpoint_path
    print(f"Loading model from: {checkpoint_path}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        str(checkpoint_path),
        load_in_4bit=cfg.training.load_in_4bit,
        use_gradient_checkpointing=cfg.training.use_gradient_checkpointing,
        max_seq_length=cfg.training.max_seq_length
    )
    
    # Set model to inference mode
    FastVisionModel.for_inference(model)
    
    # Initialize WandB for logging evaluation results
    wandb.init(
        project=cfg.wandb.project, 
        name=f"eval_{checkpoint_path.parent.name}",
        dir=project_root / cfg.out.path,
        config=OmegaConf.to_container(cfg)
    )
    
    # Prepare test dataset
    test_dataset_converted = [to_inference_conversation(d, SYSTEM_TEXT, USER_TEXT,
                                                         img_size=tuple(cfg.training.img_size),
                                                         img_color_padding=tuple(cfg.training.img_color_padding))
                              for d in tqdm(test_ds)]
    
    # Run inference on test data
    results = []
    print("Running inference on test data...")
    
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
            'label_intolerance': labels['label_intolerance'],
            'label_incivility': labels['label_incivility'],
            'label_hateful': labels['label_hateful'],
            'output': output,
        })
    
    # Extract true labels and predictions
    y_true_incivility = [r['label_incivility'] for r in results]
    y_true_intolerance = [r['label_intolerance'] for r in results]
    
    y_pred = [extract_multi_labels(r['output']) for r in results]
    y_pred_incivility = [pred[0] for pred in y_pred]
    y_pred_intolerance = [pred[1] for pred in y_pred]
    
    # Get valid predictions only
    valid_incivility = [i for i, pred in enumerate(y_pred_incivility) if pred != -1]
    y_true_incivility_valid = [y_true_incivility[i] for i in valid_incivility]
    y_pred_incivility_valid = [y_pred_incivility[i] for i in valid_incivility]
    
    valid_intolerance = [i for i, pred in enumerate(y_pred_intolerance) if pred != -1]
    y_true_intolerance_valid = [y_true_intolerance[i] for i in valid_intolerance]
    y_pred_intolerance_valid = [y_pred_intolerance[i] for i in valid_intolerance]
    
    print(f"Valid incivility predictions: {len(y_true_incivility_valid)}/{len(y_true_incivility)} ({len(y_true_incivility_valid)/len(y_true_incivility)*100:.1f}%)")
    print(f"Valid intolerance predictions: {len(y_true_intolerance_valid)}/{len(y_true_intolerance)} ({len(y_true_intolerance_valid)/len(y_true_intolerance)*100:.1f}%)")
    
    # Evaluate the predictions
    evaluation_incivility = binary_evaluation(y_true_incivility, y_pred_incivility)
    evaluation_intolerance = binary_evaluation(y_true_intolerance, y_pred_intolerance)
    
    # Calculate average metrics
    avg_accuracy = (evaluation_incivility['accuracy'] + evaluation_intolerance['accuracy']) / 2
    avg_f1 = (evaluation_incivility['f1_score'] + evaluation_intolerance['f1_score']) / 2
    avg_invalid_prediction_rate = (evaluation_incivility['invalid_prediction_rate'] + evaluation_intolerance['invalid_prediction_rate']) / 2
    
    # Log metrics to wandb
    wandb.log({
        'eval/accuracy': avg_accuracy,
        'eval/f1': avg_f1,
        'eval/invalid_prediction_rate': avg_invalid_prediction_rate,
        'eval/total_samples': len(y_true_incivility),
        
        'eval/incivility/invalid_prediction_rate': evaluation_incivility['invalid_prediction_rate'],
        'eval/incivility/accuracy': evaluation_incivility['accuracy'],
        'eval/incivility/precision': evaluation_incivility['precision'],
        'eval/incivility/recall': evaluation_incivility['recall'],
        'eval/incivility/f1': evaluation_incivility['f1_score'],
        'eval/incivility/valid_samples': len(y_true_incivility_valid),
        'eval/incivility/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_incivility_valid,
            preds=y_pred_incivility_valid,
            class_names=['Civil', 'Uncivil']
        ),
        
        'eval/intolerance/invalid_prediction_rate': evaluation_intolerance['invalid_prediction_rate'],
        'eval/intolerance/accuracy': evaluation_intolerance['accuracy'],
        'eval/intolerance/precision': evaluation_intolerance['precision'],
        'eval/intolerance/recall': evaluation_intolerance['recall'],
        'eval/intolerance/f1': evaluation_intolerance['f1_score'],
        'eval/intolerance/valid_samples': len(y_true_intolerance_valid),
        'eval/intolerance/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_intolerance_valid,
            preds=y_pred_intolerance_valid,
            class_names=['Tolerant', 'Intolerant']
        )
    })
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()