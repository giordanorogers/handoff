#!/bin/bash

# Run both shards in parallel, each on its own GPU
python -m scripts.hsp_step_accuracies --gpu 3 --shard 0 --num-shards 2 &
python -m scripts.hsp_step_accuracies --gpu 4 --shard 1 --num-shards 2 &

# Wait for both to complete
wait

echo "Both shards complete."