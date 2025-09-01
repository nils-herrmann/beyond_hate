# %%
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
import os
from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import random
import wandb

from beyond_hate.train.utils import binary_evaluation, extract_label, convert_to_conversation_inference, resize_and_pad, HateMemeDataset

# %%
# Load the configuration
cfg = OmegaConf.merge(OmegaConf.load("./config/default.yaml"),
                      OmegaConf.load("./config/eval.yaml"))
config = cfg.training

# %%
# Load labeled data
with open(f'{cfg.out.path}/labels.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
## Get relative paths for images
data = [{**item, 'img': '/'.join(item['img'].split('/')[-2:])} for item in data]
## Binarize labels
data = [{**item,
         'label_incivility': 1 if item['label_incivility'] > 0 else 0,
         'label_intolerance': 1 if item['label_incivility'] > 0 else 0}
         for item in data]
## Just keep items with valid image paths
data = [item for item in data if os.path.exists(f"{cfg.data.paths.hf}/{item['img']}")]

# %%
## Set random seed for reproducibility
random.seed(config.seed)

# First split: 90% train+val, 10% test
train_val_data, test_data = train_test_split(
    data, 
    test_size=0.1, 
    random_state=config.seed, 
    stratify=[item['label_hateful'] for item in data]  # Stratify by incivility
)

# Second split: 75% train, 15% val from the train_val_data
train_data, val_data = train_test_split(
    train_val_data, 
    test_size=0.176,  # 0.176 * 0.85 = 0.15 of total data
    random_state=config.seed,
    stratify=[item['label_hateful'] for item in train_val_data]
)
# %%
SYSTEM_TEXT = ("You are a content moderation assistant. Aid me to"
                " label images with text as hateful or neutral."
                " Hateful image are defined as containing a direct or indirect"
                " attack on people based on characteristics, including"
                " ethnicity, race, nationality, immigration status, religion,"
                " caste, sex, gender identity, sexual orientation, and"
                " disability or disease.")

USER_TEXT = (' Considering the image and its text: "{}".'
             ' Is the content of the image and its text hateful or neutral? '
             ' Respond only with the word "Hateful" or "Neutral".')

# Prepare the dataset
train_dataset = HateMemeDataset(train_data, SYSTEM_TEXT, USER_TEXT, cfg.data.paths.hf,
                                size=tuple(config.img_size or []), color_padding=tuple(config.img_color_padding or []))

# %%
# Get model and tokenizer
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
output_dir = f'{cfg.out.runs}/{current_time}'
wandb.init(project=cfg.wandb.project, name=current_time, dir=cfg.out.path, config=dict(config))

# %%
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

# %%
FastVisionModel.for_inference(model)

results = []
for sample in tqdm(test_data):
    label = sample['label_hateful']
    text = sample['text']
    
    # Resize and pad the image if specified in the config
    image = resize_and_pad(Image.open(f"{cfg.data.paths.hf}/{sample['img']}"), target_size=tuple(config.img_size), color=(255, 255, 255))
    
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

# Relabel
possible_labels = {'Hateful': 1, 'Neutral': 0}
y_true = [r['label'] for r in results]
y_pred = [extract_label(r['output'], possible_labels) for r in results]

evaluation = binary_evaluation(y_true, y_pred)

# Log metrics to wandb
wandb.log({
    "test/accuracy": evaluation['accuracy'],
    "test/precision": evaluation['precision'],
    "test/recall": evaluation['recall'],
    "test/f1_score": evaluation['f1_score'],
    "test/total_samples": len(y_true),
    "test/invalid_prediction_rate": evaluation['invalid_prediction_rate'],
})

# Finish wandb run
wandb.finish()