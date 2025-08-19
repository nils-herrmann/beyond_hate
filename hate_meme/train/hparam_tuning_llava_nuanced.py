from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
import os
import random
import time

from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import wandb

from hate_meme.train.utils import binary_evaluation, convert_to_conversation_inference, extract_multi_labels, MultiVariableDataset, resize_and_pad

# System prompt for multi-variable classification
SYSTEM_TEXT = ("You are a content moderation assistant. You need to classify images with text "
               "across multiple dimensions: incivility and intolerance. "
               "- Incivility: Rude, disrespectful or dismissive tone towards others as well as opinions expressed with antinormative intensity."
               "- Intolerance: Behaviors that are threatening to democracy and pluralism - such as prejudice, segregation, hateful or violent speech, and the use of stereotyping in order to disqualify others and groups.")

USER_TEXT = ('Considering the image and its text: "{}". '
            'Classify this content on two dimensions: '
            '1. Incivility: Is this content civil or uncivil? '
            '2. Intolerance: Is this content tolerant or intolerant? '
            'Respond in the format: "Incivility: [Civil/Uncivil], Intolerance: [Tolerant/Intolerant]"')

# Load the configuration
cfg = OmegaConf.load("./config/default.yaml")
nuanced_cfg = OmegaConf.load("./config/nuanced.yaml")
## Merge configurations and set the training configuration
cfg = OmegaConf.merge(cfg, nuanced_cfg)
config = cfg.training

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
train_dataset = MultiVariableDataset(train_data, SYSTEM_TEXT, USER_TEXT, cfg.data.paths.hf,
                                     size=tuple(config.img_size or []), color_padding=tuple(config.img_color_padding or []))

runs = [
    {'lora_r': 64, 'lora_alpha': 128, 'learning_rate': 2e-4, 'weight_decay': 0.1, 'num_train_epochs': 2},
]

for run in tqdm(runs):
    ## Merge configurations and set the training configuration
    cfg = OmegaConf.merge(OmegaConf.load("./config/default.yaml"),
                          OmegaConf.load("./config/nuanced.yaml"))
    config = cfg.training

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
    output_dir = f'{cfg.out.runs}/{current_time}'
    wandb.init(project=cfg.wandb.project, name=current_time, dir=cfg.out.path, config=dict(config))

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

    # Inference
    FastVisionModel.for_inference(model)

    results = []
    #for sample in tqdm(val_data):
    for sample in tqdm(test_data):
        label_intolerance = sample['label_intolerance']
        label_incivil = sample['label_incivility']
        label_hateful = sample['label_hateful']
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

    # Evaluate the predictions
    evaluation_incivil = binary_evaluation(y_true_incivil, y_pred_incivil)
    evaluation_intolerance = binary_evaluation(y_true_intolerance, y_pred_intolerance)

    # Calculate average metrics
    avg_accuracy = (evaluation_incivil['accuracy'] + evaluation_intolerance['accuracy']) / 2
    avg_f1 = (evaluation_incivil['f1_score'] + evaluation_intolerance['f1_score']) / 2
    avg_invalid_prediction_rate = (evaluation_incivil['invalid_prediction_rate'] + evaluation_intolerance['invalid_prediction_rate']) / 2

    wandb.log({
        'test/accuracy': avg_accuracy,
        'test/f1': avg_f1,

        'test/invalid_prediction_rate': avg_invalid_prediction_rate,
        'test/incivil/invalid_prediction_rate': evaluation_incivil['invalid_prediction_rate'],
        'test/incivil/accuracy': evaluation_incivil['accuracy'],
        'test/incivil/precision': evaluation_incivil['precision'],
        'test/incivil/recall': evaluation_incivil['recall'],
        'test/incivil/f1': evaluation_incivil['f1_score'],
        'test/incivil/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_incivil,
            preds=y_pred_incivil,
            class_names=['Civil', 'Uncivil']
        ),

        'test/intolerance/invalid_prediction_rate': evaluation_intolerance['invalid_prediction_rate'],
        'test/intolerance/accuracy': evaluation_intolerance['accuracy'],
        'test/intolerance/precision': evaluation_intolerance['precision'],
        'test/intolerance/recall': evaluation_intolerance['recall'],
        'test/intolerance/f1': evaluation_intolerance['f1_score'],
        'test/intolerance/confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_intolerance,
            preds=y_pred_intolerance,
            class_names=['Tolerant', 'Intolerant']
        )
    })

    wandb.finish()