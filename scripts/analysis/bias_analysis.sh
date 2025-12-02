#!/bin/bash
set -e

# Run the bias analysis script
poetry run python beyond_hate/analysis/bias_analysis.py
