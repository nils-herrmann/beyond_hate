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

from beyond_hate.train.prompts import coarse_prompt
from beyond_hate.train.utils import extract_label, binary_evaluation
from beyond_hate.eval.utils import get_reasoning_and_output, create_conversation
from beyond_hate.logger import get_logger


async def process_sample_async(client, sample, system_text, user_text, img_size, img_color_padding, model):
    """Process a single sample asynchronously."""
    text = sample["text"]
    image = sample["image"]
    label = sample["label_hateful"]
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
        'label': label,
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
    config_coarse_path = project_root / 'config/coarse.yaml'

    # Load configurations
    cfg = OmegaConf.load(config_base_path)
    custom_cfg = OmegaConf.load(config_coarse_path)

    # Override default config with custom config
    cfg = OmegaConf.merge(cfg, custom_cfg)

    # Load logger
    logs_dir = project_root / cfg.out.logs
    logger = get_logger("eval_coarse_gpt", logs_dir=logs_dir)

    logger.info("Starting coarse-grained GPT evaluation...")

    # Create results directory if it doesn't exist
    results_dir = project_root / cfg.out.results / 'gpt'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the test data
    logger.info(f"Loading dataset: {cfg.data.final_dataset}")
    test_ds = load_dataset(cfg.data.final_dataset, split='test')
    logger.info("Loaded test samples")

    # Load prompts
    SYSTEM_TEXT = coarse_prompt['system']
    USER_TEXT = coarse_prompt['user']

    # Run evaluation on test set
    logger.info(f"Running evaluation using model: {cfg.gpt.model}")

    img_size = tuple(cfg.training.img_size)
    img_color_padding = tuple(cfg.training.img_color_padding)

    # Run async evaluation with concurrent requests
    results = asyncio.run(run_async_evaluation(test_ds, SYSTEM_TEXT, USER_TEXT, img_size, img_color_padding, cfg.gpt.model, max_concurrent=cfg.gpt.concurrent_requests))

    logger.info(f"Evaluation complete! Processed {len(results)} samples.")

    # Calculate metrics
    possible_labels = {'Hateful': 1, 'Neutral': 0}
    y_true = [r['label'] for r in results]
    y_pred = [extract_label(r['output'], possible_labels) for r in results]

    valid = [i for i, pred in enumerate(y_pred) if pred != -1]

    # Evaluate predictions
    evaluation = binary_evaluation(y_true, y_pred)

    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {evaluation['accuracy']:.4f}")
    logger.info(f"  Precision: {evaluation['precision']:.4f}")
    logger.info(f"  Recall: {evaluation['recall']:.4f}")
    logger.info(f"  F1 Score: {evaluation['f1_score']:.4f}")
    logger.info(f"  Invalid Prediction Rate: {evaluation['invalid_prediction_rate']:.4f}")

    # Save results to file
    timestamp = time.strftime("%y%m%d-%H%M%S")
    results_file = results_dir / f"eval_{cfg.gpt.model}_coarse_{timestamp}.jsonl"

    # Save individual results
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Results saved to: {results_file}")

    # Save evaluation metrics
    metrics_file = results_dir / f"eval_{cfg.gpt.model}_coarse_{timestamp}_metrics.json"
    metrics = {
        'model': cfg.gpt.model,
        'total_samples': len(y_true),
        'valid_samples': len(valid),
        'accuracy': float(evaluation['accuracy']),
        'precision': float(evaluation['precision']),
        'recall': float(evaluation['recall']),
        'f1_score': float(evaluation['f1_score']),
        'invalid_prediction_rate': float(evaluation['invalid_prediction_rate']),
        'confusion_matrix': evaluation['confusion_matrix'].tolist()
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
