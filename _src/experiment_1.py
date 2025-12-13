"""
experiment_1.py
"""

import argparse
import json
import os
import random
from _src.handoff import HandoffEngine
from _src.text_utils import split_into_steps

def main(args):
    # Load data
    traces = []
    with open(args.input_file, 'r') as f:
        for line in f:
            traces.append(json.loads(line))

    print(f"Loaded {len(traces)} correct traces.")

    # Initialize junior
    engine = HandoffEngine(model_path=args.junior_model)

    results = []

    for item in traces:
        problem = item['problem']
        gt = item['ground_truth']
        original_trace = item['senior_trace']

        # Create shuffled trace (incoherent baseline)
        steps = split_into_steps(original_trace)
        if len(steps) < 3: continue # Skip very short traces

        # Keep first and last step, shuffle middle
        middle_steps = steps[1:-1]
        random.shuffle(middle_steps)
        shuffled_trace = "\n".join([steps[0]] + middle_steps + [steps[-1]])

        print(f"\nProcessing ID {item['id']}...")

        # Run handoff on original (coherent)
        print("     Running Coherent Handoff...")
        hsr_coherent = engine.compute_hsr(problem, gt, original_trace)

        # Run Handoff on Shuffled (incoherent)
        print("     Running Incoherent Handoff...")
        hsr_incoherent = engine.compute_hsr(problem, gt, shuffled_trace)

        results.append({
            "id": item['id'],
            "hsr_coherent": hsr_coherent,
            "hsr_incoherent": hsr_incoherent
        })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Experiment 1 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--junior_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)