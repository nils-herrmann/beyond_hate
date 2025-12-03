#!/bin/bash
set -e

# Run GPT-based fine-grained evaluation
poetry run python -m beyond_hate.eval.eval_fine_gpt

