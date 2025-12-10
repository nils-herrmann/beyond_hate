from dotenv import load_dotenv
load_dotenv()

from unsloth import is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import time
import wandb

from datasets import load_dataset
from omegaconf import OmegaConf
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm


from beyond_hate.train.utils import binary_evaluation, extract_multi_labels, to_inference_conversation ,to_train_conversation_multilabel
from beyond_hate.train.prompts import fine_prompt
from beyond_hate.logger import get_logger

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
    config = cfg.training

    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("train_fine", logs_dir=logs_dir)

    logger.info("Starting fine-grained training...")

    # Data paths
    hf_path = project_root / cfg.data.paths.hf

    # Define system and user text from prompts
    SYSTEM_TEXT = fine_prompt['system']
    USER_TEXT = fine_prompt['user']


    # Load the data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    train_ds = load_dataset(cfg.data.final_dataset, split='train')
    val_ds = load_dataset(cfg.data.final_dataset, split='validation')
    logger.info(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples")


    #%% Load the runs configuration
    runs = OmegaConf.to_container(cfg.runs)
    logger.info(f"Running {len(runs)} training configuration(s)")

    for run in tqdm(runs):
        ## Merge configurations and set the training configuration
        config = cfg.training.copy()

        ## Update the configuration with the current run parameters
        for h_param, value in run.items():
            config[h_param] = value
            logger.info(f"  {h_param}: {value}")

        # Get data
        logger.info("Preparing training dataset...")
        train_multilabel_converted = [to_train_conversation_multilabel(d, SYSTEM_TEXT, USER_TEXT, img_size=tuple(cfg.training.img_size), img_color_padding=tuple(cfg.training.img_color_padding))
                                      for d in tqdm(train_ds)]

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
        output_dir = project_root / cfg.out.runs / current_time
        logger.info(f"Initializing WandB run: {current_time}")
        wandb.init(project=cfg.wandb.project, name=current_time, dir=project_root / cfg.out.path, config=dict(config))

        # Train the model
        logger.info("Starting training...")
        FastVisionModel.for_training(model)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            data_collator = UnslothVisionDataCollator(model, tokenizer),
            train_dataset = train_multilabel_converted,
            args = SFTConfig(
                per_device_train_batch_size = config.per_device_train_batch_size,
                gradient_accumulation_steps = config.gradient_accumulation_steps,
                warmup_steps = config.warmup_steps,
                num_train_epochs = config.num_train_epochs,
                #max_steps = 10,
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

        # Inference
        logger.info("Starting evaluation on validation set...")
        FastVisionModel.for_inference(model)

        val_dataset_converted = [to_inference_conversation(d, SYSTEM_TEXT, USER_TEXT)
                                 for d in tqdm(val_ds)]

        results = []
        for conversation, image, data_id, labels in tqdm(val_dataset_converted):

            prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = tokenizer(images=image, text=prompt, return_tensors="pt").to("cuda:0")

            # autoregressively complete prompt
            max_new_tokens = 50
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            output = output.split('[/INST]')[-1]

            results.append(
                {
                    'id': data_id,
                    'label_intolerance': labels['label_intolerance'],
                    'label_incivility': labels['label_incivility'],
                    'label_hateful': labels['label_hateful'],
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

        logger.info(f"Valid incivility predictions: {len(y_true_incivil_valid)}/{len(y_true_incivil)} ({len(y_true_incivil_valid)/len(y_true_incivil)*100:.1f}%)")
        logger.info(f"Valid intolerance predictions: {len(y_true_intolerance_valid)}/{len(y_true_intolerance)} ({len(y_true_intolerance_valid)/len(y_true_intolerance)*100:.1f}%)")

        # Evaluate the predictions
        evaluation_incivil = binary_evaluation(y_true_incivil, y_pred_incivil)
        evaluation_intolerance = binary_evaluation(y_true_intolerance, y_pred_intolerance)

        # Calculate average metrics
        avg_accuracy = (evaluation_incivil['accuracy'] + evaluation_intolerance['accuracy']) / 2
        avg_f1 = (evaluation_incivil['f1_score'] + evaluation_intolerance['f1_score']) / 2
        avg_invalid_prediction_rate = (evaluation_incivil['invalid_prediction_rate'] + evaluation_intolerance['invalid_prediction_rate']) / 2
        
        logger.info("Validation Results:")
        logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Average F1: {avg_f1:.4f}")
        logger.info(f"  Incivility - Accuracy: {evaluation_incivil['accuracy']:.4f}, F1: {evaluation_incivil['f1_score']:.4f}")
        logger.info(f"  Intolerance - Accuracy: {evaluation_intolerance['accuracy']:.4f}, F1: {evaluation_intolerance['f1_score']:.4f}")

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

        logger.info(f"Run {run_idx} completed!\n")
        wandb.finish()

if __name__ == "__main__":
    main()