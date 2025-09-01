from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
import os
import random
import time

from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import wandb

from beyond_hate.train.utils import binary_evaluation, convert_to_conversation_inference, extract_multi_labels, MultiVariableDataset, resize_and_pad
from beyond_hate.train.prompts import fine_prompt

# Config paths
project_root = Path(__file__).parent.parent.parent.resolve()

config_base_path = project_root / 'config/default.yaml'
config_fine_path = project_root / 'config/fine.yaml'

# Load configurations
cfg = OmegaConf.load(config_base_path)
custom_cfg = OmegaConf.load(config_fine_path)

# Override default config with custom config
cfg = OmegaConf.merge(cfg, custom_cfg)
config = cfg.training

# Data paths
hf_path = project_root / cfg.data.paths.hf
labels_file = project_root / cfg.data.paths.labels_file

# Define system and user text from prompts
SYSTEM_TEXT = fine_prompt['system']
USER_TEXT = fine_prompt['user']

# Load labeled data
with open(labels_file, 'r') as f:
    data = [json.loads(line) for line in f]
## Get relative paths for images
data = [{**item, 'img': '/'.join(item['img'].split('/')[-2:])} for item in data]
## Binarize labels
data = [{**item,
         'label_incivility': 1 if item['label_incivility'] > 0 else 0,
         'label_intolerance': 1 if item['label_incivility'] > 0 else 0}
         for item in data]
## Just keep items with valid image paths
data = [item for item in data if os.path.exists(hf_path / item['img'])]

## Set random seed for reproducibility
random.seed(config.seed)

# First split: 85% train+val, 15% test
train_val_data, test_data = train_test_split(
    data, 
    test_size=0.1, 
    random_state=config.seed, 
    stratify=[item['label_hateful'] for item in data]  # Stratify by incivility
)

# Second split: 85% train, 15% val from the train_val_data
train_data, val_data = train_test_split(
    train_val_data, 
    test_size=0.176,  # 0.176 * 0.85 = 0.15 of total data
    random_state=config.seed,
    stratify=[item['label_hateful'] for item in train_val_data]
)

# Split the data into training and validation sets
train_dataset = MultiVariableDataset(train_data, SYSTEM_TEXT, USER_TEXT, hf_path,
                                     size=tuple(config.img_size or []), color_padding=tuple(config.img_color_padding or []))

#%% Load the runs configuration
runs = OmegaConf.to_container(cfg.runs)

for run in tqdm(runs):
    ## Merge configurations and set the training configuration
    config = cfg.training.copy()

    ## Update the configuration with the current run parameters
    for h_param, value in run.items():
        config[h_param] = value

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
    wandb.init(project=cfg.wandb.project, name=current_time, dir=project_root / cfg.out.path, config=dict(config))

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
            #num_train_epochs = config.num_train_epochs,
            max_steps = 10,
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

    # Inference
    FastVisionModel.for_inference(model)

    results = []
    val_data = val_data[:30]
    for sample in tqdm(val_data):
        label_intolerance = sample['label_intolerance']
        label_incivil = sample['label_incivility']
        label_hateful = sample['label_hateful']
        text = sample['text']
        
        # Resize and pad the image if specified in the config
        image = resize_and_pad(Image.open(hf_path/sample['img']), target_size=tuple(config.img_size), color=(255, 255, 255))
        
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
                'label_intolerance': label_intolerance,
                'label_incivility': label_incivil,
                'label_hateful': label_hateful,
                'output': output,
            }
        )

    # Extract true labels and predictions
    y_true_incivil = [r['label_incivility'] for r in results]
    y_true_intolerance = [r['label_intolerance'] for r in results]

    y_pred = [extract_multi_labels(r['output']) for r in results]
    y_pred_incivil = [pred[0] for pred in y_pred]
    y_pred_intolerance = [pred[1] for pred in y_pred]

    # Get valid predictions only
    valid_incivil = [i for i, pred in enumerate(y_pred_incivil) if pred != -1]
    y_true_incivil_valid = [y_true_incivil[i] for i in valid_incivil]
    y_pred_incivil_valid = [y_pred_incivil[i] for i in valid_incivil]

    valid_intolerance = [i for i, pred in enumerate(y_pred_intolerance) if pred != -1]
    y_true_intolerance_valid = [y_true_intolerance[i] for i in valid_intolerance]
    y_pred_intolerance_valid = [y_pred_intolerance[i] for i in valid_intolerance]

    # Evaluate the predictions
    evaluation_incivil = binary_evaluation(y_true_incivil, y_pred_incivil)
    evaluation_intolerance = binary_evaluation(y_true_intolerance, y_pred_intolerance)

    # Calculate average metrics
    avg_accuracy = (evaluation_incivil['accuracy'] + evaluation_intolerance['accuracy']) / 2
    avg_f1 = (evaluation_incivil['f1_score'] + evaluation_intolerance['f1_score']) / 2
    avg_invalid_prediction_rate = (evaluation_incivil['invalid_prediction_rate'] + evaluation_intolerance['invalid_prediction_rate']) / 2

    wandb.log({
        'val/accuracy': avg_accuracy,
        'val/f1': avg_f1,

        'val/invalid_prediction_rate': avg_invalid_prediction_rate,
        'val/incivil/invalid_prediction_rate': evaluation_incivil['invalid_prediction_rate'],
        'val/incivil/accuracy': evaluation_incivil['accuracy'],
        'val/incivil/precision': evaluation_incivil['precision'],
        'val/incivil/recall': evaluation_incivil['recall'],
        'val/incivil/f1': evaluation_incivil['f1_score'],
        'val/incivil/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_incivil_valid,
            preds=y_pred_incivil_valid,
            class_names=['Civil', 'Uncivil']
        ),

        'val/intolerance/invalid_prediction_rate': evaluation_intolerance['invalid_prediction_rate'],
        'val/intolerance/accuracy': evaluation_intolerance['accuracy'],
        'val/intolerance/precision': evaluation_intolerance['precision'],
        'val/intolerance/recall': evaluation_intolerance['recall'],
        'val/intolerance/f1': evaluation_intolerance['f1_score'],
        'val/intolerance/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_intolerance_valid,
            preds=y_pred_intolerance_valid,
            class_names=['Tolerant', 'Intolerant']
        )
    })

    wandb.finish()