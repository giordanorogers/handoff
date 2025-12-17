import os
import gc
import re
import torch
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

SEED = 9001

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
TENSOR_PARALLEL_SIZE_SENIOR = 2
TENSOR_PARALLEL_SIZE_JUNIOR = 1

N_SAMPLES = 20

MODEL_SENIOR = "Qwen/Qwen3-32B"
MODEL_JUNIOR = "Qwen/Qwen3-4B"

DATASET_NAME = "nlile/hendrycks-MATH-benchmark"
OUTPUT_FILE = "data/filtered_math_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

def run_inference(
    model_path,
    tp_size,
    problems,
    solutions,
    is_reasoning,
    want_correct,
    max_tokens=31_000,
):
    print(f"\n--- Loading {model_path} (TP={tp_size}) ---")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=False
    )
    tokenizer = llm.get_tokenizer()
    
    keep_flags = []
    
    for i, (prob, sol) in enumerate(zip(problems, solutions)):
        
        if not is_reasoning:
            
            user_content = f"Problem: {prob}\n\nOutput ONLY the final answer inside \\boxed{{}}."
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # Apply template but do not generate the prompt yet
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            
            # Manually append tokens for "\boxed{"
            prompt_text = tokenizer.decode(prompt_ids) + "\\boxed"
            
        else:
            
            # CoT Mode
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
        # 3. Adjust Stop Tokens
        # If we forced "\boxed{", we need to accept that the model might just output "5}" and stop.
        limit = max_tokens if is_reasoning else 25
        
        params = SamplingParams(
            n=N_SAMPLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=limit,
            seed=SEED
        )
        outputs = llm.generate([prompt_text], params, use_tqdm=True)
        completions = [o.text for o in outputs[0].outputs]
        
        # Verify
        correct_count = 0
        for comp in completions:
            if not is_reasoning:
                full_answer = "\\boxed" + comp
            else:
                full_answer = comp    
                
            if MathVerifier.is_correct(full_answer, sol):
                correct_count += 1
                
        accuracy = correct_count / N_SAMPLES
        print(f"  Acc={accuracy:.2f}")
        
        if want_correct:
            keep = (accuracy == 1.0)  # Keep if all correct
        else:
            keep = (accuracy == 0.0)  # Keep if all incorrect

        keep_flags.append(keep)
        
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--- Unloaded {model_path} ---")

    return keep_flags

def filter_dataset():
    # Load initial data
    ds = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(ds)[3:]
    print(f"Initial pool: {len(df)} problems")
    
    # PHASE 1: Junior Model
    # M_jun must be INCORRECT with Direct and CoT
    
    # Step 1.1: Junior Direct
    # If junior gets it right directly, it's too easy. Discard.
    print("\nPhase 1.1: Checking Junior Direct (Expect Failure)")
    junior_direct_keep = run_inference(
        MODEL_JUNIOR,
        TENSOR_PARALLEL_SIZE_JUNIOR,
        df['problem'].tolist(),
        df['solution'].tolist(),
        is_reasoning=False,
        want_correct=False
    )
            
    df = df.iloc[junior_direct_keep].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Junior Failed Direct)")
    if len(df) == 0: return
    
    # Step 1.2: Junior CoT
    print("\nPhase 1.2: Checking Junior CoT (Expect Failure)")
    junior_cot_keep = run_inference(
        MODEL_JUNIOR,
        TENSOR_PARALLEL_SIZE_JUNIOR,
        df["problem"].tolist(),
        df['solution'].tolist(),
        is_reasoning=True,
        want_correct=False,
    )
            
    df = df.iloc[junior_cot_keep].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Junior Failed CoT)")
    if len(df) == 0: return
    
    # PHASE 2: SENIOR MODEL (M_sen)
    # Goal: Keep problems where M_sen FAILS Direct but SUCCEEDS with CoT
    
    # 2.1 Senior Direct
    print("\nPhase 2.1: Checking Senior Direct (Expect Failure)")
    senior_direct_keep = run_inference(
        MODEL_SENIOR,
        TENSOR_PARALLEL_SIZE_SENIOR,
        df['problem'].tolist(),
        df['solution'].tolist(),
        is_reasoning=False,
        want_correct=False
    )

    df = df.iloc[senior_direct_keep].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Senior Failed Direct)")
    if len(df) == 0: return
    
    # 2.2 Senior CoT
    senior_cot_keep = run_inference(
        MODEL_SENIOR,
        TENSOR_PARALLEL_SIZE_SENIOR,
        df["problem"].tolist(),
        df['solution'].tolist(),
        is_reasoning=True,
        want_correct=True,
    )
            
    df = df.iloc[senior_cot_keep].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Senior Succeeded CoT)")
    if len(df) == 0: return
    
    print(f"-> Final Dataset Size: {len(df)} problems")

    # SAVE
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_json(OUTPUT_FILE, orient='records', lines=True)
    print("Done.")
    
if __name__ == "__main__":
    filter_dataset()
