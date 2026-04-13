#!/usr/bin/env bash
set -e

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running main pipeline..."
python pipeline.py