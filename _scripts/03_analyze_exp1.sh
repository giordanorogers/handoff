#!/bin/bash

# Ensure dependencies are installed
pip install matplotlib seaborn pandas scipy

INPUT_DIR="results/exp1_sanity"

echo "Running Analysis for Experiment 1..."
python -m src.analyze_exp1 --input_dir $INPUT_DIR

echo "Analysis complete. Check $INPUT_DIR for plots."