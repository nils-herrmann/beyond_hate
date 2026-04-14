import os
import re

from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def slice_dataset_stratified(dataset, share_samples, label_column='label_hateful', seed=42):
    """Slice a dataset using stratified sampling to maintain label distribution.
    
    Args:
        dataset: HuggingFace dataset to slice
        share_samples: Float between 0 and 1 indicating fraction of data to keep (e.g., 0.5 for 50%)
        label_column: Name of the label column to use for stratification
        seed: Random seed for reproducibility
    
    Returns:
        Sliced dataset maintaining label distribution
    """
    if share_samples is None or share_samples >= 1.0:
        return dataset
    
    # Get stratification labels
    labels = [d[label_column] for d in dataset]
    
    # Perform stratified split
    indices = list(range(len(dataset)))
    kept_indices, _ = train_test_split(
        indices,
        test_size=1 - share_samples,
        random_state=seed,
        stratify=labels
    )
    
    # Create subset dataset
    return dataset.select(sorted(kept_indices))


def to_train_conversation(sample, system_text, user_text, img_size=(512, 512), img_color_padding=(255, 255, 255)):
    """Convert a dataset sample to conversation format for binary classification training.
    
    Args:
        sample: Dataset sample with 'image', 'label_hateful', and 'text' keys
        system_text: System prompt text
        user_text: User prompt template with placeholder for meme text
        img_size: Target image size as (width, height)
        img_color_padding: RGB color for padding as (R, G, B)
    
    Returns:
        Dictionary with 'messages' key containing the conversation structure
    """
    # Process image (already a PIL Image from HF dataset)
    image = sample['image']
    if img_size:
        image = resize_and_pad(image, target_size=img_size, color=img_color_padding)
    
    # Get label
    label = sample['label_hateful']
    label_text = 'Hateful' if label == 1 else 'Neutral'
    
    # Meme text
    meme_text = sample['text']
    user_input = user_text.format(meme_text)

    # Create conversation
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_input}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label_text}]
        },
    ]

    return {"messages": conversation}


def to_train_conversation_multilabel(sample, system_text, user_text, img_size=(512, 512), img_color_padding=(255, 255, 255)):
    """Convert a dataset sample to conversation format for multi-label classification training.
    
    Args:
        sample: Dataset sample with 'image', 'label_incivility', 'label_intolerance', and 'text' keys
        system_text: System prompt text
        user_text: User prompt template with placeholder for meme text
        img_size: Target image size as (width, height)
        img_color_padding: RGB color for padding as (R, G, B)
    
    Returns:
        Dictionary with 'messages' key containing the conversation structure
    """
    image = sample['image']
    if img_size:
        image = resize_and_pad(image, target_size=img_size, color=img_color_padding)
    
    incivility_label = 'Uncivil' if sample['label_incivility'] == 1 else 'Civil'
    intolerance_label = 'Intolerant' if sample['label_intolerance'] == 1 else 'Tolerant'

    meme_text = sample['text']
    response = f"Incivility: {incivility_label}, Intolerance: {intolerance_label}"

    conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text.format(meme_text)}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            },
        ]

    return {"messages": conversation}


def to_inference_conversation(sample, system_text, uster_text, img_size=(512, 512), img_color_padding=(255, 255, 255)):
    """Convert a dataset sample to conversation format for inference.
    
    Args:
        sample: Dataset sample with 'id', 'image', 'text', and label fields
        system_text: System prompt text
        uster_text: User prompt template with placeholder for meme text
        img_size: Target image size as (width, height)
        img_color_padding: RGB color for padding as (R, G, B)
    
    Returns:
        Tuple of (conversation, image, data_id, labels)
    """
    data_id = sample['id']
    labels = {k: v for k, v in sample.items() if k.startswith("label_")}

    # Process image (already a PIL Image from HF dataset)
    image = sample['image']
    if img_size:
        image = resize_and_pad(image, target_size=img_size, color=img_color_padding)

    meme_text = sample['text']
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_text}
            ]
        },
        {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": uster_text.format(meme_text)}
            ]
        }
    ]
    return conversation, image, data_id, labels


def extract_label(text: str, labels: dict):
    """Extract predicted label from model output text.
    
    Args:
        text: Raw output text from the vision language model
        labels: Dictionary mapping label strings to integer values
    
    Returns:
        Integer label value if found, -1 otherwise
    """
    text_clean = re.sub(r'\s+', ' ', text.lower()).strip()
    for label, label_int in labels.items():
        if label.lower() in text_clean:
            return label_int
    return -1


def resize_and_pad(image, target_size=(256, 256), color=(255, 255, 255)):
    """Resize image to fit within target size while maintaining aspect ratio and pad to exact size.
    
    Args:
        image: PIL Image object
        target_size: Target dimensions as (width, height)
        color: RGB color for padding as (R, G, B)
    
    Returns:
        Resized and padded PIL Image
    """
    image = image.convert('RGB')
    
    # Resize while maintaining aspect ratio
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Calculate padding
    delta_w = target_size[0] - image.size[0]
    delta_h = target_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # Add padding
    padded_image = ImageOps.expand(image, padding, fill=color)
    return padded_image

def extract_multi_labels(text: str):
    """Extract both incivility and intolerance labels from model output.
    
    Args:
        text: Raw output text from the model
    
    Returns:
        Tuple of (incivility_label, intolerance_label) where each is 0, 1, or -1 (invalid)
    """
    text_clean = text.lower().strip()
    
    # Extract incivility
    incivility = -1
    if 'incivility: civil' in text_clean or 'incivility:civil' in text_clean:
        incivility = 0
    elif 'incivility: uncivil' in text_clean or 'incivility:uncivil' in text_clean:
        incivility = 1
    
    # Extract intolerance
    intolerance = -1
    if 'intolerance: tolerant' in text_clean or 'intolerance:tolerant' in text_clean:
        intolerance = 0
    elif 'intolerance: intolerant' in text_clean or 'intolerance:intolerant' in text_clean:
        intolerance = 1
    
    return incivility, intolerance

def to_train_conversation_joint(sample, system_text, user_text, img_size=(512, 512), img_color_padding=(255, 255, 255)):
    """Convert a dataset sample to conversation format for joint classification training (incivility, intolerance, hatefulness).
    
    Args:
        sample: Dataset sample with 'image', 'label_incivility', 'label_intolerance', 'label_hateful', and 'text' keys
        system_text: System prompt text
        user_text: User prompt template with placeholder for meme text
        img_size: Target image size as (width, height)
        img_color_padding: RGB color for padding as (R, G, B)
    
    Returns:
        Dictionary with 'messages' key containing the conversation structure
    """
    image = sample['image']
    if img_size:
        image = resize_and_pad(image, target_size=img_size, color=img_color_padding)
    
    incivility_label = 'Uncivil' if sample['label_incivility'] == 1 else 'Civil'
    intolerance_label = 'Intolerant' if sample['label_intolerance'] == 1 else 'Tolerant'
    hateful_label = 'Hateful' if sample['label_hateful'] == 1 else 'Neutral'

    meme_text = sample['text']
    response = f"Incivility: {incivility_label}, Intolerance: {intolerance_label}, Hatefulness: {hateful_label}"

    conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text.format(meme_text)}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            },
        ]

    return {"messages": conversation}


def extract_joint_labels(text: str):
    """Extract incivility, intolerance, and hatefulness labels from model output.
    
    Args:
        text: Raw output text from the model
    
    Returns:
        Tuple of (incivility_label, intolerance_label, hateful_label) where each is 0, 1, or -1 (invalid)
    """
    text_clean = text.lower().strip()
    
    # Extract incivility
    incivility = -1
    if 'incivility: civil' in text_clean or 'incivility:civil' in text_clean:
        incivility = 0
    elif 'incivility: uncivil' in text_clean or 'incivility:uncivil' in text_clean:
        incivility = 1
    
    # Extract intolerance
    intolerance = -1
    if 'intolerance: tolerant' in text_clean or 'intolerance:tolerant' in text_clean:
        intolerance = 0
    elif 'intolerance: intolerant' in text_clean or 'intolerance:intolerant' in text_clean:
        intolerance = 1
    
    # Extract hatefulness
    hateful = -1
    if 'hatefulness: neutral' in text_clean or 'hatefulness:neutral' in text_clean:
        hateful = 0
    elif 'hatefulness: hateful' in text_clean or 'hatefulness:hateful' in text_clean:
        hateful = 1
    
    return incivility, intolerance, hateful


def binary_evaluation(y_true, y_pred):
    """Calculate evaluation metrics for binary classification predictions.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
    
    Returns:
        Dictionary with metrics: invalid_prediction_rate, accuracy, precision, recall, f1_score, confusion_matrix
    """
    valid = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_pred_valid = [y_pred[i] for i in valid]

    if len(valid) > 0:
        return {
            'invalid_prediction_rate': (len(y_pred) - len(y_pred_valid)) / len(y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred, normalize='true')
        }
    else:
        print("No valid predictions to evaluate.")
        return {
                'invalid_prediction_rate': (len(y_pred) - len(y_pred_valid)) / len(y_pred),
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'confusion_matrix': None
            }