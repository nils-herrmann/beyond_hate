# Beyond Hate Project

This project advances multimodal toxic speech detection by distinguishing between tone (incivility) and content (intolerance) in hateful meme classification. We propose a fine-grained annotation scheme that moves beyond binary hateful/not-hateful labels, enabling more nuanced and accurate content moderation through improved data quality and model training.

## Paper
The full paper is available in [BeyondHate.pdf](./BeyondHate.pdf).

## Installation guide
### 1. Clone repository
```bash
git clone https://github.com/nils-herrmann/beyond_hate.git
cd beyond_hate
```

### 2. Install dependencies (with Poetry)
#### 2.1 Install [Poetry](https://python-poetry.org/docs/#installation) (with pip) if you haven't already:
   ```bash
   pip install poetry
   ```

#### 2.2.a Install minimal project dependencies for annotaiton
   ```bash
   poetry install --without=dev
   ```
#### 2.2.b Intall all project dependencies
   ```bash
   poetry install
   ```

#### 2.3 To find the interpreter path for Poetry, run:
   ```bash
   poetry run which python
   ```

### 3. Download data (with HuggingFace Hub)
#### 3.1 Set `HF_TOKEN` in [.env](.env) file. You can get your token from [HuggingFace](https://huggingface.co/settings/tokens).

#### 3.2 Execute the data download script:
   ```bash
   poetry run python beyond_hate/data_processing/download_hateful_meme_hf.py
   ```

### 4. Fine-tune LLaVA
#### 4.1 Set `WAND_API_KEY` in [.env](.env) file. You can get your key from [Weights & Biases](https://wandb.ai/authorize).

#### 4.2 Fine-tune LLaVA for single-label classification (coarse):
   ```bash
   poetry run python beyond_hate/train/train_coarse.py
   ```

#### 4.3 Fine-tune LLaVA for multi-label classification (fine grained):
   ```bash
   poetry run python beyond_hate/train/train_fine.py
   ```

### 5. Evaluate model
#### 5.1 Set `evaluation` `checkpoint_path` in [coarse.yaml](./config/coarse.yaml) and [fine.yaml](./config/fine.yaml).

#### 5.2 Evaluate coarse model:
   ```bash
   poetry run python beyond_hate/eval/eval_coarse.py
   ```

#### 5.3 Evaluate fine-grained model:
   ```bash
   poetry run python beyond_hate/eval/eval_fine.py
   ```