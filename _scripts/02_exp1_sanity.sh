#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

INPUT_FILE="data/senior_traces/math_lvl5.jsonl"
OUTPUT_DIR="results/exp1_sanity"

# Create a small subset of correct traces for the experiment
# We only want traces where is_correct is true
grep '"is_correct": true' $INPUT_FILE | head -n 20 > data/senior_traces/exp1_subset.jsonl

python -m src.experiment_1 \
    --input_file "data/senior_traces/exp1_subset.jsonl" \
    --junior_model "Qwen/Qwen3-8B" \
    --output_dir $OUTPUT_DIR