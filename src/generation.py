import os
import json
import random
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
from src.text_utils import (
    check_correctness,
    split_into_steps,
    apply_chat_template
)

def main(args):
    # 1. Load Dataset
    print(f"Loading GSM8K ({args.split})...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    # 2. Initialize Senior Model
    print(f"Initializing Senior Model: {args.senior_model}")
    # Run across 2 GPUs
    llm_senior = LLM(model=args.senior_model, tensor_parallel_size=args.senior_gpus, trust_remote_code=True)
    tokenizer = llm_senior.get_tokenizer()

    # Sampling params: Greedy for "Correct", High Temp for "Incorrect" generation
    params_greedy = SamplingParams(temperature=0.0, max_tokens=30_000)
    params_stochastic = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=30_000)

    # 3. Batch Generation for Senior (Greedy)
    prompts = [apply_chat_template(tokenizer, q) for q in dataset['questions']]
    
    print("Generating Senior traces (Greedy)...")
    senior_outputs = llm_senior.generate(prompts, params_greedy)

    # 4. Filter for Senior Correctness
    candidates = []
    for i, output in enumerate(senior_outputs):
        pred_text = output.outputs[0].text
        gt_text = dataset['answer'][i]

        if check_correctness(pred_text, gt_text):
            candidates.append({
                "question": dataset['question'][i],
                "ground_truth": gt_text,
                "senior_correct_trace": pred_text
            })

    print(f"Senior solved {len(candidates)}/{len(dataset)} problems.")

    # Clean up Senior LLM to free memory for Junior
    import gc
    import torch
    del llm_senior
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Filter for Junior Failure (The "Hard" Subset)
    print(f"Initializing Junior Model for filtering: {args.junior_model}")
    llm_junior = LLM(model=args.junior_model, tensor_parallel_size=1, trust_remote_code=True)

    # Prepare prompts for candidates
    cand_prompts = [apply_chat_template(llm_junior.get_tokenizer(), c['question']) for c in candidates]

    print("Running Junior on candidates...")
    junior_outputs = llm_junior.generate(cand_prompts, params_greedy)

    filtered_data = []
    for i, output in enumerate(junior_outputs):
        pred_text = output.outputs[0].text
        if not check_correctness(pred_text, candidates[i]['ground_truth']):
            # Senior Correct AND Junior Incorrect -> Keep
            filtered_data.append(candidates[i])

    print(f"Filtered down to {len(filtered_data)} problems where Senior > Junior.")

    # Clean up Junior
    del llm_junior
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Generate Variations (Incorrect & Shuffled)
    print("Re-initializing Senior to generate 'Incorrect' variations...")
    llm_senior = LLM(model=args.senior_model, tensor_parallel_size=args.senior_gpus, trust_remote_code=True)
    
    final_output = []

    # Only process a subset for Exp 1 if dataset is huge
    target_problems = filtered_data[:args.limit] if args.limit else filtered_data

    # Prepare batch for incorrect generation (stochastic sampling)
    incorrect_prompts = [apply_chat_template(tokenizer, item['question']) for item in target_problems]

    # Generate 4 samples per problem to increase odds of finding a wrong one
    params_n4 = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=30_000, n=4)
    incorrect_results = llm_senior.generate(incorrect_prompts, params_n4)

    for i, item in enumerate(target_problems):
        question = item['question']
        gt = item['ground_truth']
        correct_trace = item['senior_correct_trace']

        # A. Get Incorrect trace
        incorrect_trace = None
        for output in incorrect_results[i].outputs:
            if not check_correctness(output.text, gt):
                incorrect_trace = output.text
                break

        # Fallback: if senior got all 4 right (too smart!), skip this item.
        if incorrect_trace is None:
            continue

        # B. Get Shuffled Trace
        steps = split_into_steps(correct_trace)
        random.shuffle(steps)
        shuffled_trace = "\n".join(steps)

        final_output.append({
            "id": i,
            "question": question,
            "ground_truth": gt,
            "traces": {
                "correct": correct_trace,
                "incorrect": incorrect_trace,
                "shuffled": shuffled_trace
            }
        })

    # 7. Save to JSONL
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for entry in final_output:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(final_output)} processed traces to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--senior_model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--junior_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--senior_gpus", type=int, default=2)
    parser.add_argument("--split", type=str, default="test") # Use 'test' for quick validation, 'train' for full exp
    parser.add_argument("--limit", type=int, default=100) # Number of final traces needed for Exp 1
    parser.add_argument("--output_file", type=str, default="data/senior_traces/exp1_sanity.jsonl")
    args = parser.parse_args()
    main(args)
