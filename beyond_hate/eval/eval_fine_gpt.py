from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import time

from datasets import load_dataset
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.auto import tqdm

from beyond_hate.train.prompts import fine_prompt
from beyond_hate.train.utils import extract_multi_labels, binary_evaluation
from beyond_hate.eval.utils import get_reasoning_and_output, create_conversation
from beyond_hate.logger import get_logger


async def process_sample_async(client, sample, system_text, user_text, img_size, img_color_padding, model):
    """Process a single sample asynchronously."""
    text = sample["text"]
    image = sample["image"]
    label_incivility = sample["label_incivility"]
    label_intolerance = sample["label_intolerance"]
    label_hateful = sample["label_hateful"]
    data_id = sample["id"]

    # Create conversation and get prediction
    conversation = create_conversation(text, image, system_text, user_text, img_size, img_color_padding)

    response = await client.responses.create(
        model=model,
        input=conversation
    )

    reasoning, output = get_reasoning_and_output(response)

    return {
        'id': data_id,
        'label_incivility': label_incivility,
        'label_intolerance': label_intolerance,
        'label_hateful': label_hateful,
        'output': output,
        'reasoning': reasoning,
    }


async def run_async_evaluation(test_ds, system_text, user_text, img_size, img_color_padding, model, max_concurrent=10):
    """Run evaluation with concurrent API requests."""
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(sample):
        async with semaphore:
            return await process_sample_async(client, sample, system_text, user_text, img_size, img_color_padding, model)
    
    # Create tasks for all samples
    tasks = [process_with_semaphore(sample) for sample in test_ds]
    
    # Run with progress bar
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples"):
        result = await coro
        results.append(result)
    
    return results


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

    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("eval_fine_gpt", logs_dir=logs_dir)

    logger.info("Starting fine-grained GPT evaluation...")

    # Create results directory if it doesn't exist
    results_dir = project_root / cfg.out.results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the test data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    test_ds = load_dataset(cfg.data.final_dataset, split='test')
    logger.info("Loaded test samples")

    # Define system and user text from prompts
    SYSTEM_TEXT = fine_prompt['system']
    USER_TEXT = fine_prompt['user']

    # Run evaluation on test set
    logger.info(f"Running evaluation using model: {cfg.gpt.model}")

    img_size = tuple(cfg.training.img_size)
    img_color_padding = tuple(cfg.training.img_color_padding)

    # Run async evaluation with concurrent requests
    results = asyncio.run(run_async_evaluation(test_ds, SYSTEM_TEXT, USER_TEXT, img_size, img_color_padding, cfg.gpt.model, max_concurrent=cfg.gpt.concurrent_requests))

    logger.info(f"Evaluation complete! Processed {len(results)} samples.")

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

    logger.info(f"Valid incivility predictions: {len(y_true_incivility_valid)}/{len(y_true_incivility)} ({len(y_true_incivility_valid)/len(y_true_incivility)*100:.1f}%)")
    logger.info(f"Valid intolerance predictions: {len(y_true_intolerance_valid)}/{len(y_true_intolerance)} ({len(y_true_intolerance_valid)/len(y_true_intolerance)*100:.1f}%)")

    # Evaluate the predictions
    evaluation_incivility = binary_evaluation(y_true_incivility, y_pred_incivility)
    evaluation_intolerance = binary_evaluation(y_true_intolerance, y_pred_intolerance)

    # Calculate average metrics
    avg_accuracy = (evaluation_incivility['accuracy'] + evaluation_intolerance['accuracy']) / 2
    avg_f1 = (evaluation_incivility['f1_score'] + evaluation_intolerance['f1_score']) / 2
    avg_invalid_prediction_rate = (evaluation_incivility['invalid_prediction_rate'] + evaluation_intolerance['invalid_prediction_rate']) / 2

    logger.info("Evaluation Results:")
    logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"  Average F1: {avg_f1:.4f}")
    logger.info(f"  Incivility - Accuracy: {evaluation_incivility['accuracy']:.4f}, F1: {evaluation_incivility['f1_score']:.4f}")
    logger.info(f"  Intolerance - Accuracy: {evaluation_intolerance['accuracy']:.4f}, F1: {evaluation_intolerance['f1_score']:.4f}")

    # Save results to file
    timestamp = time.strftime("%y%m%d-%H%M%S")
    results_file = results_dir / f"eval_{cfg.gpt.model}_fine_{timestamp}.jsonl"

    # Save individual results
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Results saved to: {results_file}")

    # Save evaluation metrics
    metrics_file = results_dir / f"eval_{cfg.gpt.model}_fine_{timestamp}_metrics.json"
    metrics = {
        'model': cfg.gpt.model,
        'total_samples': len(y_true_incivility),
        'valid_incivility_samples': len(valid_incivility),
        'valid_intolerance_samples': len(valid_intolerance),
        'avg_accuracy': float(avg_accuracy),
        'avg_f1': float(avg_f1),
        'avg_invalid_prediction_rate': float(avg_invalid_prediction_rate),
        'incivility': {
            'accuracy': float(evaluation_incivility['accuracy']),
            'precision': float(evaluation_incivility['precision']),
            'recall': float(evaluation_incivility['recall']),
            'f1_score': float(evaluation_incivility['f1_score']),
            'invalid_prediction_rate': float(evaluation_incivility['invalid_prediction_rate']),
            'confusion_matrix': evaluation_incivility['confusion_matrix'].tolist()
        },
        'intolerance': {
            'accuracy': float(evaluation_intolerance['accuracy']),
            'precision': float(evaluation_intolerance['precision']),
            'recall': float(evaluation_intolerance['recall']),
            'f1_score': float(evaluation_intolerance['f1_score']),
            'invalid_prediction_rate': float(evaluation_intolerance['invalid_prediction_rate']),
            'confusion_matrix': evaluation_intolerance['confusion_matrix'].tolist()
        }
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
