import os
import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
from _src.text_utils import check_correctness, apply_chat_template

def main(args):
    # -------------------------------------------------------------------------
    # 1. Load and Filter Dataset (MATH)
    # -------------------------------------------------------------------------
    print(f"Loading MATH dataset (hendrycks/competition_math)...")
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="train", trust_remote_code=True)
    
    print(f"Total problems before filtering: {len(dataset)}")
    
    # Filter for Level 5 (Hardest) to get a good mix of Correct/Incorrect
    # You can also add 'Level 4' if you need more volume: 
    # lambda x: x['level'] in ['Level 5', 'Level 4']
    dataset = dataset.filter(lambda x: x['level'] == 5)
    
    # Optional: Shuffle and limit to avoid processing 10k problems for a test run
    if args.limit:
        dataset = dataset.shuffle(seed=42).select(range(args.limit))
        
    print(f"Selected {len(dataset)} Level 5 problems for generation.")

    # -------------------------------------------------------------------------
    # 2. Initialize Senior Model
    # -------------------------------------------------------------------------
    print(f"Initializing Senior Model: {args.senior_model}")
    
    # CRITICAL: Set max_model_len to 32768 as discussed.
    # This prevents OOM on long reasoning while allowing deep CoT.
    llm_senior = LLM(
        model=args.senior_model, 
        tensor_parallel_size=args.senior_gpus, 
        max_model_len=32768, 
        trust_remote_code=True,
        gpu_memory_utilization=0.95 # Slightly aggressive memory usage to fit context
    )
    tokenizer = llm_senior.get_tokenizer()

    # -------------------------------------------------------------------------
    # 3. Sampling Parameters
    # -------------------------------------------------------------------------
    # We use Greedy (temp=0) to find "Natural" failures (failures at best effort).
    # max_tokens=12288 ensures we don't cut off valid long reasoning (approx 12k tokens).
    # We don't go to 30k output to save time on infinite loops.
    params_greedy = SamplingParams(
        temperature=0.0, 
        max_tokens=12288,
        stop=["<|im_end|>", "<|endoftext|>"] # Ensure clean stops
    )

    # -------------------------------------------------------------------------
    # 4. Generate Traces
    # -------------------------------------------------------------------------
    print("Preparing prompts...")
    # MATH dataset uses 'problem' key
    prompts = [apply_chat_template(tokenizer, p) for p in dataset['problem']]
    
    print(f"Generating traces for {len(prompts)} problems...")
    senior_outputs = llm_senior.generate(prompts, params_greedy)
    
    # -------------------------------------------------------------------------
    # 5. Process and Classify
    # -------------------------------------------------------------------------
    results = []
    correct_count = 0
    length_failure_count = 0
    
    for i, output in enumerate(senior_outputs):
        pred_text = output.outputs[0].text
        gt_text = dataset['solution'][i]
        problem_text = dataset['problem'][i]
        finish_reason = output.outputs[0].finish_reason
        
        # Check for length failure (non-convergent reasoning)
        if finish_reason == "length":
            length_failure_count += 1
            # We explicitly mark length failures as incorrect
            is_correct = False
        else:
            # Use our robust tex_utils checker
            is_correct = check_correctness(pred_text, gt_text)
        
        if is_correct:
            correct_count += 1
            
        results.append({
            "id": i,
            "problem": problem_text,
            "ground_truth": gt_text,
            "senior_trace": pred_text,
            "is_correct": is_correct,
            "finish_reason": finish_reason,
            "token_count": len(output.outputs[0].token_ids)
        })

    # -------------------------------------------------------------------------
    # 6. Summary and Save
    # -------------------------------------------------------------------------
    accuracy = correct_count / len(results)
    print(f"\n--- Generation Summary ---")
    print(f"Total Traces: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Length Failures (>12k tokens): {length_failure_count}")
    
    # Save to JSONL
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved traces to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--senior_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--senior_gpus", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for testing")
    parser.add_argument("--output_file", type=str, default="data/senior_traces/math_lvl5.jsonl")
    args = parser.parse_args()
    main(args)