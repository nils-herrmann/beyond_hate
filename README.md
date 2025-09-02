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
2.1 Install [Poetry](https://python-poetry.org/docs/#installation) (with pip) if you haven't already:
   ```bash
   pip install poetry
   ```

2.2.a Install minimal project dependencies for annotaiton
   ```bash
   poetry install --without=dev
   ```
2.2.b Intall all project dependencies
   ```bash
   poetry install
   ```

2.3 To find the interpreter path for Poetry, run:
   ```bash
   poetry run which python
   ```

### 3. Download data (with HuggingFace Hub)
3.1 Set `HF_TOKEN` in [.env](.env) file. You can get your token from [HuggingFace](https://huggingface.co/settings/tokens).

3.2 Execute the data download script:
   ```bash
   poetry run python beyond_hate/data_processing/download_hateful_meme_hf.py
   ```

### 4. Train LLaVA
4.1 Set `WAND_API_KEY` in [.env](.env) file. You can get your key from [Weights & Biases](https://wandb.ai/authorize).

4.2 Fine-tune LLaVA for single-label classification (coarse):
   ```bash
   poetry run python beyond_hate/train/train_coarse.py
   ```

4.3 Fine-tune LLaVA for multi-label classification (fine grained):
   ```bash
   poetry run python beyond_hate/train/train_fine.py
   ```

### 5. Evaluate model


## Repository Structure

```
.
├── .gitignore             # Git ignore rules
├── dockerfile             # Docker configuration to build image with requirements
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── runpod_setup.sh       # RunPod environment setup script
│
├── config/                # Configuration files
│   ├── default.yaml       # Default training configuration
│   ├── eval.yaml         # Evaluation configuration
│   └── nuanced.yaml      # Fine-grained annotation configuration
│
├── data/                  # Data storage (gitignored)
│   └── hateful_memes_hf/  # Hateful Memes dataset from HuggingFace
│       ├── train.jsonl
│       ├── dev_seen.jsonl
│       ├── dev_unseen.jsonl
│       ├── test_seen.jsonl
│       ├── test_unseen.jsonl
│       └── img/           # Image files
│
├── beyond_hate/             # Main package source code
│   ├── __init__.py
│   ├── analysis/          # Analysis and evaluation scripts
│   │   └── bias_analysis.ipynb # False-positive, false-negative analysis
│   ├── data_processing/   # Data handling and preprocessing
│   │   ├── __init__.py
│   │   ├── hateful_meme_annotation.ipynb # Annotate hateful meme dataset using ipwdigets
│   │   ├── hateful_meme_downoad_validation # Validate hateful meme dataset from huggingface and kaggle
│   │   ├── annotate.py    # Manual annotation helpers
│   │   └── download_hateful_meme_hf.py  # Data download utilities
│   └── train/             # Model training scripts
│       ├── __init__.py
│       ├── utils.py       # Training utilities and data processing functions
│       ├── finetune_llava_hateful.py          # Fine-tune LLaVA for binary hateful/neutral classification
│       ├── finetune_llava_hateful.ipynb       # Notebook version of binary classification training
│       ├── finetune_llava_hateful_600.py      # Fine-tune LLaVA on 600-sample subset for binary classification
│       ├── finetune_llava_nuanced.ipynb       # Fine-tune LLaVA for nuanced incivility/intolerance classification
│       ├── hparam_tuning_llava_hateful.py     # Hyperparameter tuning for binary classification model
│       ├── hparam_tuning_llava_nuanced.py     # Hyperparameter tuning for nuanced classification model
│       ├── wandb_tuning_llava_nuanced.py      # Weights & Biases sweep for nuanced model hyperparameter optimization
│       ├── inference_llava_hateful_600.py     # Run inference with 600-sample trained model
│       ├── inference_llava_nuanced.ipynb      # Run inference with nuanced classification model
│       └── eval_llava.ipynb                   # Evaluate model performance and analyze misclassifications
│
├── out/                   # Output directory (gitignored)
│   ├── metrics/           # Training metrics and logs
│   ├── runs/              # Model checkpoints and results
│   └── predictions/       # Model predictions
│
└── wandb/                 # Weights & Biases logs (gitignored)
```