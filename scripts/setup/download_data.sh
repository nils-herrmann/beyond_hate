#!/bin/bash
set -e

# Activate poetry environment and run the script
poetry run python ../../hate_meme/data_processing/download_hateful_meme_hf.py
