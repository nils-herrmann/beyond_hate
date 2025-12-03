#!/bin/bash
set -e

# Download the data
poetry run python -m beyond_hate.data_processing.download_hateful_meme_hf
