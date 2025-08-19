# %%
from dotenv import load_dotenv
load_dotenv()

from unsloth import FastVisionModel

import json
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm.notebook import tqdm

from hate_meme.train.utils import extract_label, convert_to_conversation_inference, resize_and_pad

# %%
# Load the configuration
cfg = OmegaConf.merge(OmegaConf.load("../../config/default.yaml"),
                      OmegaConf.load("../../config/eval.yaml"))
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
# Get model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    config.model,
    load_in_4bit = config.load_in_4bit,
    use_gradient_checkpointing = config.use_gradient_checkpointing, 
    max_seq_length = config.max_seq_length
)
# %%
FastVisionModel.for_inference(model)

results = []
for sample in tqdm(data):
    label = sample['label_hateful']
    text = sample['text']
    
    # Resize and pad the image if specified in the config
    image = resize_and_pad(Image.open(f"{cfg.data.paths.hf}/{sample['img']}"), target_size=tuple(config.img_size), color=(255, 255, 255))
    #image = unsloth_preprocessing(Image.open(f"{hf_path}/{sample['img']}"), model)
    #image = Image.open(f"{hf_path}/{sample['img']}")
    
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

# %%
# Save results
import pandas as pd

df = pd.DataFrame(results)
df['y_true'] = y_true
df['y_pred'] = y_pred
df.to_csv("/workspace/disinfo/out/runs/250708-1815/results.csv", index=False)

# %%
