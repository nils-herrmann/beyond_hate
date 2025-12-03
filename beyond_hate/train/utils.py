import os
import re

from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset


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


class MultiVariableDataset(Dataset):
    """Dataset for multi-label classification with incivility and intolerance labels.
    
    Args:
        data: List of data samples
        system_text: System prompt text
        user_text: User prompt template
        image_base_path: Base directory path for images
        size: Target image size as (width, height)
        color_padding: RGB color for padding as (R, G, B)
    """
    def __init__(self, data, system_text, user_text, image_base_path, size=(256, 266), color_padding=(255, 255, 255)):
        self.data = data
        self.system_text = system_text
        self.user_text = user_text
        self.image_base_path = image_base_path
        self.size = size
        self.color_padding = color_padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.copy()
        sample['img'] = os.path.join(self.image_base_path, sample['img'])
        image = Image.open(sample['img'])
        if self.size:
            image = resize_and_pad(image, target_size=self.size, color=self.color_padding)
        
        return self.convert_to_conversation_train(sample, image)

    def convert_to_conversation_train(self, sample, image):
        """Convert sample to conversation format for training.
        
        Args:
            sample: Data sample with label fields
            image: Processed PIL Image
        
        Returns:
            Dictionary with 'messages' key containing conversation structure
        """
        # Assuming your data has labels like 'label_incivility' and 'label_intolerance'
        incivility_label = 'Uncivil' if sample.get('label_incivility', 0) == 1 else 'Civil'
        intolerance_label = 'Intolerant' if sample.get('label_intolerance', 0) == 1 else 'Tolerant'
        
        meme_text = sample['text']
        response = f"Incivility: {incivility_label}, Intolerance: {intolerance_label}"

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_text.format(meme_text)}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            },
        ]
        return {"messages": conversation}

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

def binary_evaluation(y_true, y_pred):
    """Calculate evaluation metrics for binary classification predictions.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels (may contain -1 for invalid predictions)
    
    Returns:
        Dictionary with metrics: invalid_prediction_rate, accuracy, precision, recall, f1_score, confusion_matrix
    """
    valid = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true_valid = [y_true[i] for i in valid]
    y_pred_valid = [y_pred[i] for i in valid]

    return {
        'invalid_prediction_rate': (len(y_pred) - len(y_true_valid)) / len(y_pred),
        'accuracy': accuracy_score(y_true_valid, y_pred_valid),
        'precision': precision_score(y_true_valid, y_pred_valid, average='weighted'),
        'recall': recall_score(y_true_valid, y_pred_valid, average='weighted'),
        'f1_score': f1_score(y_true_valid, y_pred_valid, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true_valid, y_pred_valid, normalize='true')
    }
