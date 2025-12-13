import json
from text_utils import check_correctness

input_file = "data/senior_traces/math_lvl5.jsonl"
output_file = "data/senior_traces/math_lvl5_corrected.jsonl"

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        data = json.loads(line)
        # Re-check correctness
        is_correct = check_correctness(data['problem'], data['senior_trace'], data['ground_truth'])
        data['is_correct'] = is_correct
        fout.write(json.dumps(data) + '\n')

print("Re-evaluation complete. Use 'math_lvl5_corrected.jsonl' for Experiment 1.")