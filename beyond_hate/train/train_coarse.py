#%%
from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import time
import torch
import wandb

from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm


from beyond_hate.train.utils import binary_evaluation, extract_label, to_train_conversation, to_inference_conversation, slice_dataset_stratified
from beyond_hate.train.prompts import coarse_prompt
from beyond_hate.logger import get_logger


def main():
    # Config paths
    project_root = Path(__file__).parent.parent.parent.resolve()

    config_base_path = project_root / 'config/default.yaml'
    config_coarse_path = project_root / 'config/coarse.yaml'
    config_runs_path = project_root / 'config/runs.yaml'
    
    # Load configurations
    cfg = OmegaConf.load(config_base_path)
    runs_cfg = OmegaConf.load(config_runs_path)
    custom_cfg = OmegaConf.load(config_coarse_path)

    # Override default config with custom config
    cfg = OmegaConf.merge(cfg, runs_cfg, custom_cfg)

    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("train_coarse", logs_dir=logs_dir)

    logger.info("Starting coarse-grained training...")

    # Load the data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    train_ds = load_dataset(cfg.data.final_dataset, split='train')
    val_ds = load_dataset(cfg.data.final_dataset, split='validation')
    logger.info(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples")

    # Load prompts
    SYSTEM_TEXT = coarse_prompt['system']
    USER_TEXT = coarse_prompt['user']

    #%% Load the runs configuration

    runs = OmegaConf.to_container(cfg.runs)
    logger.info(f"Running {len(runs)} training configuration(s)")

    #%%
    for run_idx, run in enumerate(tqdm(runs)):
        # Load the default training configuration
        config = cfg.training.copy()
        
        # Update the configuration with the current run parameters
        for h_param, value in run.items():
            config[h_param] = value

        # Apply stratified sampling if share_samples is specified
        train_ds_subset = train_ds
        if 'share_samples' in run and run['share_samples'] is not None:
            share = run['share_samples']
            logger.info(f"Applying stratified sampling with share_samples={share}")
            train_ds_subset = slice_dataset_stratified(train_ds, share, seed=config.get('seed', 42))
            logger.info(f"Dataset sliced from {len(train_ds)} to {len(train_ds_subset)} samples")

        # Prepare the dataset
        logger.info("Preparing training dataset...")
        train_dataset_converted = [to_train_conversation(d, SYSTEM_TEXT, USER_TEXT, img_size=tuple(cfg.training.img_size), img_color_padding=tuple(cfg.training.img_color_padding))
                                   for d in tqdm(train_ds_subset)]

        # Load the model and tokenizer
        logger.info(f"Loading model: {config.model}")
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
        output_dir = project_root / cfg.out.runs / "coarse" / current_time
        logger.info(f"Initializing WandB run: {current_time}")
        wandb.init(project=cfg.wandb.project, name=current_time, dir=project_root/cfg.out.path, config=dict(config))
        run_id = wandb.run.id

        # Train the model
        logger.info("Starting training...")
        FastVisionModel.for_training(model)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            data_collator = UnslothVisionDataCollator(model, tokenizer),
            train_dataset = train_dataset_converted,
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
        logger.info("Training completed!")

        # Evaluate the model
        logger.info("Starting evaluation on validation set...")
        FastVisionModel.for_inference(model)
        val_dataset_converted = [to_inference_conversation(d, SYSTEM_TEXT, USER_TEXT)
                                 for d in tqdm(val_ds)]

        batch_size = cfg.validation.batch_size
        results = []
        # Process in batches
        for i in tqdm(range(0, len(val_dataset_converted), batch_size), desc="Evaluating batches"):
            batch = val_dataset_converted[i:i+batch_size]
            
            # Prepare batch data
            conversations = [item[0] for item in batch]
            images = [item[1] for item in batch]
            data_ids = [item[2] for item in batch]
            labels_list = [item[3] for item in batch]
            
            # Apply chat template and tokenize batch
            prompts = [tokenizer.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
            inputs = tokenizer(images=images, text=prompts, return_tensors="pt", padding=True).to("cuda:0")

            # Generate for batch
            max_new_tokens = 50
            input_length = inputs['input_ids'].shape[1]
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
            # Extract only generated tokens (model-agnostic approach)
            generated_tokens = outputs[:, input_length:]
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Process each output in the batch
            for data_id, labels, decoded_output in zip(data_ids, labels_list, decoded_outputs):
                output = decoded_output.strip()
                results.append(
                    {
                        'id': data_id,
                        'label': labels['label_hateful'],
                        'output': output,
                    }
                )

        # Calculate metrics
        possible_labels = {'Hateful': 1, 'Neutral': 0}

        y_true = [r['label'] for r in results]
        y_pred = [extract_label(r['output'], possible_labels) for r in results]
        
        # Evaluate
        evaluation = binary_evaluation(y_true, y_pred)
        
        logger.info(f"Validation Results:")
        logger.info(f"  Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation['precision']:.4f}")
        logger.info(f"  Recall: {evaluation['recall']:.4f}")
        logger.info(f"  F1 Score: {evaluation['f1_score']:.4f}")
        logger.info(f"  Invalid Prediction Rate: {evaluation['invalid_prediction_rate']:.4f}")

        # Reinit run if finished
        if wandb.run is None:
            wandb.init(project=cfg.wandb.project, id=run_id, name=current_time, dir=project_root/cfg.out.path, resume="allow")

        # Log metrics to wandb
        wandb.log({
            "val/hateful/accuracy": evaluation['accuracy'],
            "val/hateful/precision": evaluation['precision'],
            "val/hateful/recall": evaluation['recall'],
            "val/hateful/f1_score": evaluation['f1_score'],
            "val/hateful/total_samples": len(y_true),
            "val/hateful/invalid_prediction_rate": evaluation['invalid_prediction_rate'],
            "val/hateful/confusion_matrix": evaluation['confusion_matrix'].tolist(),
        })

        # Finish wandb run
        logger.info(f"Run {run_idx} completed!\n")
        wandb.finish()

if __name__ == "__main__":
    main()