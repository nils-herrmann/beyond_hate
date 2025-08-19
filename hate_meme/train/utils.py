import os
from PIL import Image, ImageOps
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset

class HateMemeDataset(Dataset):
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
        # Open the image and resize it if necessary
        image = Image.open(sample['img'])
        if self.size:
            image = resize_and_pad(image, target_size=self.size, color=self.color_padding)
    
        return convert_to_conversation_train(sample, self.system_text, self.user_text, image)


def convert_to_conversation_inference(meme_text, system_text, uster_text):
    """Convert a sample from the Hate Meme dataset into a conversation format for inference.
    Args:
        meme_text (str): The text of the meme.
        system_text (str): The system message to be included in the conversation.
        uster_text (str): The user message template to be filled with the meme text."""
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
    return conversation


def convert_to_conversation_train(sample, system_text, uster_text, image):
  """
  Convert a sample from the Hate Meme dataset into a conversation format.
    Args:
        sample (dict): A sample from the Hate Meme dataset containing 'img', 'label_hateful' (or label), and 'text'.
        system_text (str): The system message to be included in the conversation.
        uster_text (str): The user message template to be filled with the meme text.
  """
  label = sample.get('label') or sample.get('label_hateful')
  label = 'Hateful' if label == 1 else 'Neutral'
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
            {"type": "image", "image": image},
            {"type": "text", "text": uster_text.format(meme_text)}
          ],
      },
      {"role": "assistant",
       "content": [
           {"type": "text", "text": label}
       ]},
  ]
  return {"messages": conversation}


def extract_label(text: str, labels: dict):
    """
    Post processing the raw output from VLM. Extracting the predictions.

    Parameters:
    - text: raw output from VLM
    """
    text_clean = re.sub(r'\s+', ' ', text.lower()).strip()
    for label, label_int in labels.items():
        if label.lower() in text_clean:
            return label_int
    return -1


def resize_and_pad(image, target_size=(256, 266), color=(255, 255, 255)):
    """ Resize an image to fit within target_size while maintaining aspect ratio,
    and pad it with a specified color if necessary.
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
    """Extract both incivility and intolerance labels from model output"""
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
    """Evaluate predictions against true labels."""
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
