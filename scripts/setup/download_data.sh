#!/bin/bash
set -e

# Activate poetry environment and run the script
poetry run python ../../beyond_hate/data_processing/download_hateful_meme_hf.py
