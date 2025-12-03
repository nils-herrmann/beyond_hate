#!/bin/bash
set -e

# Run the annotation validation script
poetry run python -m beyond_hate.analysis.validate_annotations
