#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SENIOR="Qwen/Qwen3-32B"
JUNIOR="Qwen/Qwen3-8B"

python -m src.generation \
    --senior_model $SENIOR \
    --junior_model $JUNIOR \
    --senior_gpus 2 \
    --split "test" \
    --limit 100 \
    --output_file "data/senior_traces/exp1_sanity.jsonl"