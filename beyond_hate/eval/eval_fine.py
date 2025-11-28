from dotenv import load_dotenv
load_dotenv()

from unsloth import FastVisionModel
import json
import os
import random
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, convert_to_conversation_inference, extract_multi_labels, resize_and_pad
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
    
    # Data paths
    hf_path = project_root / cfg.data.paths.hf
    labels_file = project_root / cfg.data.paths.labels_file
    
    # Define system and user text from prompts
    SYSTEM_TEXT = fine_prompt['system']
    USER_TEXT = fine_prompt['user']
    
    # Load labeled data
    with open(labels_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Get relative paths for images
    data = [{**item, 'img': '/'.join(item['img'].split('/')[-2:])} for item in data]
    
    # Binarize labels
    data = [{**item,
             'label_incivility': 1 if int(item['label_incivility']) > 0 else 0,
             'label_intolerance': 1 if int(item['label_intolerance']) > 0 else 0}
             for item in data]
    
    # Just keep items with valid image paths
    data = [item for item in data if os.path.exists(hf_path / item['img'])]
    
    # Set random seed for reproducibility
    random.seed(cfg.training.seed)
    
    # First split: 85% train+val, 15% test
    train_val_data, test_data = train_test_split(
        data, 
        test_size=0.1, 
        random_state=cfg.training.seed, 
        stratify=[item['label_hateful'] for item in data]
    )
    
    #print(f"Loaded {len(test_data)} test samples")
    
    # Load the fine-tuned model
    checkpoint_path = project_root / cfg.evaluation.checkpoint_path
    #print(f"Loading model from: {checkpoint_path}")
    
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
    
    # Run inference on test data
    results = []
    print("Running inference on test data...")
    
    for sample in tqdm(test_data):
        label_intolerance = sample['label_intolerance']
        label_incivility = sample['label_incivility']
        label_hateful = sample['label_hateful']
        text = sample['text']
        
        # Resize and pad the image if specified in the config
        if cfg.training.img_size:
            image = resize_and_pad(
                Image.open(hf_path / sample['img']), 
                target_size=tuple(cfg.training.img_size), 
                color=tuple(cfg.training.img_color_padding)
            )
        else:
            image = Image.open(hf_path / sample['img'])
        
        conversation = convert_to_conversation_inference(text, SYSTEM_TEXT, USER_TEXT)
        
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = tokenizer(images=image, text=prompt, return_tensors="pt").to("cuda:0")
        
        # Generate prediction
        max_new_tokens = 50
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.split('[/INST]')[-1].strip()
        
        results.append({
            'id': sample['id'],
            'label_intolerance': label_intolerance,
            'label_incivility': label_incivility,
            'label_hateful': label_hateful,
            'text': text,
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