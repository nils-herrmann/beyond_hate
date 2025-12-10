#!/bin/bash
set -e

# Run the bias analysis script
poetry run python -m beyond_hate.analysis.bias_analysis
