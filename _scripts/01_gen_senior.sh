#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SENIOR="Qwen/Qwen3-32B"

python -m src.generation \
    --senior_model $SENIOR \
    --senior_gpus 2 \
    --limit 200 \
    --output_file "data/senior_traces/math_lvl5.jsonl"