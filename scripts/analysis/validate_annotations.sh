#!/bin/bash
set -e

# Run the annotation validation script
poetry run python beyond_hate/analysis/validate_annotations.py
