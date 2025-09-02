from dotenv import load_dotenv
load_dotenv()

from unsloth import FastVisionModel
import json
import os
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, extract_label, convert_to_conversation_inference, resize_and_pad
from beyond_hate.train.prompts import coarse_prompt

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
    
    # Load the test data
    hf_path = project_root / cfg.data.paths.hf
    with open(hf_path / 'test_seen.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Just keep items with valid image paths
    test_data = [item for item in test_data if os.path.exists(f"{hf_path}/{item['img']}")]
    
    # Load prompts
    SYSTEM_TEXT = coarse_prompt['system']
    USER_TEXT = coarse_prompt['user']
    
    # Load the fine-tuned model
    checkpoint_path = project_root / cfg.evaluation.checkpoint_path
    
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
    for sample in tqdm(test_data):
        label = sample['label']
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
            'label': label,
            'text': text,
            'output': output,
        })
    
    # Calculate metrics
    possible_labels = {'Hateful': 1, 'Neutral': 0}
    
    y_true = [r['label'] for r in results]
    y_pred = [extract_label(r['output'], possible_labels) for r in results]
    
    # Get valid predictions only
    valid = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true_valid = [y_true[i] for i in valid]
    y_pred_valid = [y_pred[i] for i in valid]
    
    print(f"Valid predictions: {len(y_true_valid)}/{len(y_true)} ({len(y_true_valid)/len(y_true)*100:.1f}%)")
    
    # Evaluate
    evaluation = binary_evaluation(y_true, y_pred)
        
    # Log confusion matrix to wandb
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
    wandb.finish()

if __name__ == "__main__":
    main()