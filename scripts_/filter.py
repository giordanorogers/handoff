import os
import gc
import json
import torch
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from src.utils import MathVerifier
from pathlib import Path

N_SAMPLES_QUICK = 5
N_SAMPLES_FULL = 20
BATCH_SIZE = 50

TENSOR_PARALLEL_SIZE_SENIOR = 1
TENSOR_PARALLEL_SIZE_JUNIOR = 1

MODEL_SENIOR = "Qwen/Qwen3-32B"
MODEL_JUNIOR = "Qwen/Qwen3-4B"

DATASET_NAME = "nlile/hendrycks-MATH-benchmark"
OUTPUT_FILE = "data/filtered_math_dataset.jsonl"
CHECKPOINT_FILE = "data/filtering_checkpoint.json"

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


def is_integer_answer(answer):
    """Check if answer can be cast to integer."""
    try:
        val = float(answer)
        return val == int(val)
    except (ValueError, TypeError):
        return False


def build_prompts(tokenizer, problems, is_reasoning):
    prompts = []
    for prob in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prob}
        ]
        if is_reasoning:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=True
            )
        else:
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False
            )
            prompt = tokenizer.decode(prompt_ids) + "\\boxed{"
        prompts.append(prompt)
    return prompts


def batch_inference(llm, prompts, solutions, is_reasoning, n_samples, max_tokens):
    if not prompts:
        return []
    
    params = SamplingParams(
        n=n_samples,
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=9001,
    )
    
    all_outputs = llm.generate(prompts, params, use_tqdm=True)
    
    accuracies = []
    for output, sol in zip(all_outputs, solutions):
        correct = 0
        for completion in output.outputs:
            text = completion.text
            full_answer = text if is_reasoning else "\\boxed{" + text
            if MathVerifier.is_correct(full_answer, str(sol)):
                correct += 1
        accuracies.append(correct / n_samples)
    
    return accuracies


def load_model(model_path, tp_size):
    print(f"\nLoading {model_path} (TP={tp_size})...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def unload_model(llm, model_path):
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Unloaded {model_path}")


def save_validated_problem(problem_row, output_file):
    with open(output_file, 'a') as f:
        f.write(json.dumps(problem_row) + '\n')
    print(f"  -> SAVED valid problem to {output_file}")


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_batch_start": 0, "total_found": 0}


def save_checkpoint(batch_start, total_found):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            "last_batch_start": batch_start,
            "total_found": total_found
        }, f)


def run_junior_stages(batch_df, llm_junior, tok_junior):
    """Run junior model filtering stages. Returns surviving indices."""
    questions = batch_df['problem'].tolist()
    solutions = batch_df['answer'].tolist()
    surviving = list(range(len(questions)))
    
    # Stage 1.1: Junior Direct (expect 0%)
    if surviving:
        prompts = build_prompts(tok_junior, [questions[i] for i in surviving], is_reasoning=False)
        accs = batch_inference(llm_junior, prompts, [solutions[i] for i in surviving],
                               is_reasoning=False, n_samples=N_SAMPLES_QUICK, max_tokens=25)
        surviving = [surviving[i] for i, acc in enumerate(accs) if acc == 0.0]
        print(f"    After Junior Direct: {len(surviving)} remain")
    
    # Stage 1.2: Junior CoT (expect 0%)
    if surviving:
        prompts = build_prompts(tok_junior, [questions[i] for i in surviving], is_reasoning=True)
        accs = batch_inference(llm_junior, prompts, [solutions[i] for i in surviving],
                               is_reasoning=True, n_samples=N_SAMPLES_QUICK, max_tokens=31_000)
        surviving = [surviving[i] for i, acc in enumerate(accs) if acc == 0.0]
        print(f"    After Junior CoT: {len(surviving)} remain")
    
    return surviving


def run_senior_stages(batch_df, surviving, llm_senior, tok_senior):
    """Run senior model filtering stages. Returns surviving indices."""
    questions = batch_df['problem'].tolist()
    solutions = batch_df['answer'].tolist()
    
    # Stage 2.1: Senior Direct (expect 0%)
    if surviving:
        prompts = build_prompts(tok_senior, [questions[i] for i in surviving], is_reasoning=False)
        accs = batch_inference(llm_senior, prompts, [solutions[i] for i in surviving],
                               is_reasoning=False, n_samples=N_SAMPLES_QUICK, max_tokens=25)
        surviving = [surviving[i] for i, acc in enumerate(accs) if acc == 0.0]
        print(f"    After Senior Direct: {len(surviving)} remain")
    
    # Stage 2.2: Senior CoT (expect 100%)
    if surviving:
        prompts = build_prompts(tok_senior, [questions[i] for i in surviving], is_reasoning=True)
        accs = batch_inference(llm_senior, prompts, [solutions[i] for i in surviving],
                               is_reasoning=True, n_samples=N_SAMPLES_FULL, max_tokens=31_000)
        surviving = [surviving[i] for i, acc in enumerate(accs) if acc == 1.0]
        print(f"    After Senior CoT: {len(surviving)} remain")
    
    return surviving


def run_junior_validation(batch_df, surviving, llm_junior, tok_junior):
    """Final junior validation with full samples. Returns surviving indices."""
    questions = batch_df['problem'].tolist()
    solutions = batch_df['answer'].tolist()
    
    if surviving:
        prompts = build_prompts(tok_junior, [questions[i] for i in surviving], is_reasoning=True)
        accs = batch_inference(llm_junior, prompts, [solutions[i] for i in surviving],
                               is_reasoning=True, n_samples=N_SAMPLES_FULL, max_tokens=31_000)
        surviving = [surviving[i] for i, acc in enumerate(accs) if acc == 0.0]
        print(f"    After Junior Final Validation: {len(surviving)} remain")
    
    return surviving


def collect_valid_problems(batch_df, surviving):
    """Convert surviving indices to problem dicts."""
    questions = batch_df['problem'].tolist()
    solutions = batch_df['answer'].tolist()
    indices = batch_df.index.tolist()
    
    valid_problems = []
    for i in surviving:
        valid_problems.append({
            "original_index": int(indices[i]),
            "problem": questions[i],
            "solution": solutions[i],
        })
    return valid_problems


def filter_dataset():
    # Load data
    ds = load_dataset(DATASET_NAME, split="test")
    df = pd.DataFrame(ds)
    print(f"Total problems in dataset: {len(df)}")
    
    # Pre-screen for integer answers
    integer_mask = df['answer'].apply(is_integer_answer)
    n_rejected = (~integer_mask).sum()
    df = df[integer_mask].reset_index(drop=True)
    print(f"After integer answer filter: {len(df)} problems ({n_rejected} rejected)")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    start_idx = checkpoint["last_batch_start"]
    total_found = checkpoint["total_found"]
    
    if start_idx > 0:
        print(f"Resuming from batch starting at index {start_idx}")
        print(f"Previously found {total_found} valid problems")
    
    # Collect batches and their junior-stage survivors
    batches_with_survivors = []
    
    # === PHASE 1: All Junior Stages ===
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    llm_junior, tok_junior = load_model(MODEL_JUNIOR, TENSOR_PARALLEL_SIZE_JUNIOR)
    
    for batch_start in range(start_idx, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\n=== Batch {batch_start}-{batch_end}: Junior Stages ===")
        surviving = run_junior_stages(batch_df, llm_junior, tok_junior)
        
        if surviving:
            batches_with_survivors.append((batch_start, batch_end, batch_df, surviving))
            print(f"  {len(surviving)} candidates passed junior stages")
        else:
            print(f"  No candidates survived junior stages")
    
    unload_model(llm_junior, MODEL_JUNIOR)
    
    if not batches_with_survivors:
        print("\nNo candidates passed junior stages. Exiting.")
        return
    
    # === PHASE 2: All Senior Stages ===
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    llm_senior, tok_senior = load_model(MODEL_SENIOR, TENSOR_PARALLEL_SIZE_SENIOR)
    
    batches_for_validation = []
    
    for batch_start, batch_end, batch_df, surviving in batches_with_survivors:
        print(f"\n=== Batch {batch_start}-{batch_end}: Senior Stages ===")
        surviving = run_senior_stages(batch_df, surviving, llm_senior, tok_senior)
        
        if surviving:
            batches_for_validation.append((batch_start, batch_end, batch_df, surviving))
            print(f"  {len(surviving)} candidates passed senior stages")
        else:
            print(f"  No candidates survived senior stages")
    
    unload_model(llm_senior, MODEL_SENIOR)
    
    if not batches_for_validation:
        print("\nNo candidates passed senior stages. Exiting.")
        return
    
    # === PHASE 3: Final Junior Validation ===
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    llm_junior, tok_junior = load_model(MODEL_JUNIOR, TENSOR_PARALLEL_SIZE_JUNIOR)
    
    for batch_start, batch_end, batch_df, surviving in batches_for_validation:
        print(f"\n=== Batch {batch_start}-{batch_end}: Final Validation ===")
        surviving = run_junior_validation(batch_df, surviving, llm_junior, tok_junior)
        
        if surviving:
            valid_problems = collect_valid_problems(batch_df, surviving)
            for prob in valid_problems:
                save_validated_problem(prob, OUTPUT_FILE)
                total_found += 1
                print(f"  *** Found valid problem #{total_found} (idx: {prob['original_index']})")
        
        save_checkpoint(batch_end, total_found)
    
    unload_model(llm_junior, MODEL_JUNIOR)
    
    print(f"\n=== FILTERING COMPLETE ===")
    print(f"Total valid problems: {total_found}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_dataset()