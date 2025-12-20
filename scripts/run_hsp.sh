#!/bin/bash
cd /disk/u/gio/handoff

for i in {0..3}; do
    python -m scripts.experiment_first_hsp --gpu $i --shard $i --num-shards 4 &
done
wait
echo "All shards complete"