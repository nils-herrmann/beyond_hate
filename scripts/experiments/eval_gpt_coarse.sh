#!/bin/bash
set -e

# Run GPT-based coarse-grained evaluation
poetry run python -m beyond_hate.eval.eval_coarse_gpt

