#%%
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
import os
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
import time
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, extract_label, convert_to_conversation_inference, resize_and_pad, HateMemeDataset
from beyond_hate.train.prompts import coarse_prompt

# Config paths
project_root = Path(__file__).parent.parent.parent.resolve()

config_base_path = project_root / 'config/default.yaml'
config_coarse_path = project_root / 'config/coarse.yaml'

# Load configurations
cfg = OmegaConf.load(config_base_path)
custom_cfg = OmegaConf.load(config_coarse_path)

# Override default config with custom config
cfg = OmegaConf.merge(cfg, custom_cfg)

# Load the data
# Load hate meme train set
hf_path = project_root / cfg.data.paths.hf
with open(hf_path / 'train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
# Just keep items with valid image paths
train_data = [item for item in train_data if os.path.exists(f"{hf_path}/{item['img']}")]

# Load hate meme dev set
hf_path = project_root / cfg.data.paths.hf
with open(hf_path / 'dev_seen.jsonl', 'r') as f:
    val_data = [json.loads(line) for line in f]

# Just keep items with valid image paths
val_data = [item for item in val_data if os.path.exists(f"{hf_path}/{item['img']}")]

# Load prompts
SYSTEM_TEXT = coarse_prompt['system']
USER_TEXT = coarse_prompt['user']

#%% Load the runs configuration

runs = OmegaConf.to_container(cfg.runs)

#%%
for run in tqdm(runs):
    # Load the default training configuration
    config = cfg.training.copy()
    
    # Update the configuration with the current run parameters
    for h_param, value in run.items():
        config[h_param] = value

    # Prepare the dataset
    train_dataset = HateMemeDataset(train_data, SYSTEM_TEXT, USER_TEXT, hf_path,
                                    size=tuple(config.img_size or []), color_padding=tuple(config.img_color_padding or []))

    # Load the model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model,
        load_in_4bit = config.load_in_4bit,
        use_gradient_checkpointing = config.use_gradient_checkpointing, 
        max_seq_length = config.max_seq_length
    )

    # Configure LoRA to enable finetuning
    model = FastVisionModel.get_peft_model(model,
                                        r=config.lora_r,
                                        lora_alpha=config.lora_alpha,
                                        lora_dropout=config.lora_dropout,
                                        max_seq_length=config.max_seq_length)

    # WandB setup
    current_time = time.strftime("%y%m%d-%H%M")
    output_dir = project_root / cfg.out.runs / current_time
    wandb.init(project=cfg.wandb.project, name=current_time, dir=project_root/cfg.out.path, config=dict(config))

    # Train the model
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_dataset,
        args = SFTConfig(
            per_device_train_batch_size = config.per_device_train_batch_size,
            gradient_accumulation_steps = config.gradient_accumulation_steps,
            warmup_steps = config.warmup_steps,
            num_train_epochs = config.num_train_epochs,
            learning_rate = config.learning_rate,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = config.logging_steps,
            optim = config.optim,
            weight_decay = config.weight_decay,
            lr_scheduler_type = config.lr_scheduler_type,
            seed = config.seed,
            output_dir = output_dir,
            report_to = config.report_to,

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = config.remove_unused_columns,
            dataset_text_field = config.dataset_text_field,
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = config.dataset_num_proc,
            max_seq_length = config.max_seq_length,

            # Save strategy
            save_strategy = config.save_strategy,
            save_total_limit = config.save_total_limit,
        )
    )
    trainer.train()

    
    FastVisionModel.for_inference(model)

    results = []
    for sample in tqdm(val_data):
        label = sample['label']
        text = sample['text']
        
        # Resize and pad the image if specified in the config
        if config.img_size:
            image = resize_and_pad(Image.open(hf_path / sample['img']), target_size=tuple(config.img_size), color=tuple(config.img_color_padding))
        else:
            image = Image.open(hf_path / sample['img'])
        
        conversation = convert_to_conversation_inference(text, SYSTEM_TEXT, USER_TEXT)

        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = tokenizer(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        max_new_tokens = 50
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.split('[/INST]')[-1]

        results.append(
            {
                'id': sample['id'],
                'label': label,
                'output': output,
            }
        )

    # Calculate metrics
    possible_labels = {'Hateful': 1, 'Neutral': 0}

    y_true = [r['label'] for r in results]
    y_pred = [extract_label(r['output'], possible_labels) for r in results]

    # Get valid predictions only
    valid = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true_valid = [y_true[i] for i in valid]
    y_pred_valid = [y_pred[i] for i in valid]
    
    # Evaluate
    evaluation = binary_evaluation(y_true, y_pred)

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
        "eval/invalid_prediction_rate": evaluation['invalid_prediction_rate']
    })

    # Finish wandb run
    wandb.finish()
