# %%
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
import os
from omegaconf import OmegaConf
import time
from trl import SFTTrainer, SFTConfig
from tqdm.notebook import tqdm
import wandb

# %%
cfg = OmegaConf.load("./config/default.yaml")
config = cfg.training # Load the training configuration

# %%
model, tokenizer = FastVisionModel.from_pretrained(
    config.model,
    load_in_4bit = config.load_in_4bit,
    use_gradient_checkpointing = config.use_gradient_checkpointing, 
    max_seq_length = config.max_seq_length
)

# %%
# Load hate meme train set
hf_path = cfg.data.paths.hf
with open(f'{hf_path}/train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
# Just keep items with valid image paths
train_data = [item for item in train_data if os.path.exists(f"{hf_path}/{item['img']}")]

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

# %%
from beyond_hate.train.utils import HateMemeDataset

train_dataset = HateMemeDataset(train_data, SYSTEM_TEXT, USER_TEXT, hf_path,
                                size=tuple(config.img_size or []), color_padding=tuple(config.img_color_padding or []))

# %%
# Configure LoRA to enable finetuning
model = FastVisionModel.get_peft_model(model,
                                       r=config.lora_r,
                                       lora_alpha=config.lora_alpha,
                                       max_seq_length=config.max_seq_length)

# %%
current_time = time.strftime("%y%m%d-%H%M")
output_dir = f'{cfg.out.runs}/{current_time}'
wandb.init(project=cfg.wandb.project, name=current_time, dir=cfg.out.path, config=dict(config))
    
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
    ),
)

trainer.train()

# %% [markdown]
# # Inference

# %%
from PIL import Image
from beyond_hate.train.utils import extract_label, convert_to_conversation_inference, resize_and_pad

# Load hate meme dev set
hf_path = cfg.data.paths.hf
with open(f'{hf_path}/dev_seen.jsonl', 'r') as f:
    val_data = [json.loads(line) for line in f]
# Just keep items with valid image paths
val_data = [item for item in val_data if os.path.exists(f"{hf_path}/{item['img']}")]

# %%
# model, tokenizer = FastVisionModel.from_pretrained(
#     "/workspace/disinfo/out/runs/20250627-1752/checkpoint-422",
#     load_in_4bit=config.load_in_4bit,
#     use_gradient_checkpointing=config.use_gradient_checkpointing,
#     max_seq_length=config.max_seq_length
# )

# %%
# TODO: Batch processing
FastVisionModel.for_inference(model)

results = []
for sample in tqdm(val_data):
    label = sample['label']
    text = sample['text']
    
    # Resize and pad the image if specified in the config
    if config.img_size:
        image = resize_and_pad(Image.open(f"{hf_path}/{sample['img']}"), target_size=tuple(config.img_size), color=tuple(config.img_color_padding))
    else:
        image = Image.open(f"{hf_path}/{sample['img']}")
    
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

# %%
possible_labels = {'Hateful': 1, 'Neutral': 0}

y_true = [r['label'] for r in results]
#y_true = [1 if pred == 'Hateful' else 0 for pred in y_true]

y_pred = [extract_label(r['output'], possible_labels) for r in results]

# Filter out invalid predictions (-1)
valid = [i for i, pred in enumerate(y_pred) if pred != -1]
y_true_valid = [y_true[i] for i in valid]
y_pred_valid = [y_pred[i] for i in valid]
print(f'Invalid predictions: {len(y_pred) - len(y_true_valid)}')

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

accuracy = accuracy_score(y_true_valid, y_pred_valid)
precision = precision_score(y_true_valid, y_pred_valid, average='weighted')  # or 'macro' depending on your needs
recall = recall_score(y_true_valid, y_pred_valid, average='weighted')
f1 = f1_score(y_true_valid, y_pred_valid, average='weighted')
cm = confusion_matrix(y_true_valid, y_pred_valid, normalize='true')

# Log metrics to wandb
wandb.log({
    "eval/accuracy": accuracy,
    "eval/precision": precision,
    "eval/recall": recall,
    "eval/f1_score": f1,
    "eval/total_samples": len(y_true),
    "eval/valid_predictions": len(y_true_valid),
    "eval/invalid_predictions": len(y_pred) - len(y_true_valid),
    "eval/invalid_prediction_rate": (len(y_pred) - len(y_true_valid)) / len(y_pred)
})

wandb.log({"eval/confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_true_valid,
    preds=y_pred_valid,
    class_names=['Neutral', 'Hateful']
)})

# Finish wandb run
wandb.finish()

# Save metrics to text file
metrics_text = f"""Model Evaluation Metrics
========================

Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1 Score: {f1:.3f}

Confusion Matrix (Normalized):
{cm}

Raw Confusion Matrix:
{confusion_matrix(y_true_valid, y_pred_valid)}

Additional Details:
- Total samples: {len(y_true)}
- Valid predictions: {len(y_true_valid)} (after filtering invalid predictions)
- Invalid predictions: {len(y_pred) - len(y_true_valid)}
"""

# Create output directory if it doesn't exist
os.makedirs(f'{cfg.out.path}/metrics', exist_ok=True)

# Save to file
metrics_file = os.path.join(f'{cfg.out.path}/metrics', f"{time.strftime('%Y%m%d_%H%M%S')}.txt")
with open(metrics_file, 'w') as f:
    f.write(metrics_text)

print(f"\nMetrics saved to: {metrics_file}")


