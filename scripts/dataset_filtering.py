import os
import gc
import re
import torch
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

# CONFIGURATION

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
TENSOR_PARALLEL_SIZE_SENIOR = 2
TENSOR_PARALLEL_SIZE_JUNIOR = 1

# MODELS

MODEL_SENIOR = "Qwen/Qwen3-32B"
MODEL_JUNIOR = "Qwen/Qwen3-8B"

# DATASET

DATASET_NAME = "nlile/hendrycks-MATH-benchmark"
OUTPUT_FILE = "filtered_math_dataset.jsonl"

FEW_SHOT_DIRECT = [
    {"role": "user", "content": "Problem: What is 2+2?\n\nOutput ONLY the final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{4}"},
    {"role": "user", "content": "Problem: Solve for x: 2x = 10.\n\nOutput ONLY the final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{5}"},
    {"role": "user", "content": "Problem: Calculate the area of a square with side length 3.\n\nOutput ONLY the final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{9}"}
]

PROMPT_COT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class MathVerifier:
    @staticmethod
    def extract_answer(text):
        if not text: 
            return None
            
        # Matches \boxed{...} allowing for whitespace (e.g., \boxed {5})
        start_indices = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
        
        if not start_indices:
            return None
            
        last_match_content = None
        
        for start in start_indices:
            balance = 0
            # Find the actual opening brace after \boxed
            content_start = text.find('{', start) + 1
            
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{':
                    balance += 1
                elif char == '}':
                    if balance == 0:
                        last_match_content = text[content_start:i]
                        break
                    balance -= 1
        
        return last_match_content

    @staticmethod
    def normalize_answer(text):
        if text is None: return ""
        
        # Strip whitespace
        text = text.strip().replace(" ", "")
        # Normalize LaTeX fractions
        text = text.replace(r"\dfrac", r"\frac")
        text = text.replace(r"\tfrac", r"\frac")
        # Strip \text{} wrappers
        text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
        
        return text

    @staticmethod
    def is_correct(generated_text, ground_truth):
        pred = MathVerifier.extract_answer(generated_text)
        truth = MathVerifier.extract_answer(ground_truth)
        
        if truth is None: truth = ground_truth
        
        # Debug printing for analysis (optional)
        # if pred: print(f"  Pred: {pred} | Truth: {truth}")
        
        return MathVerifier.normalize_answer(pred) == MathVerifier.normalize_answer(truth)

def run_batch_inference(
    model_path,
    tp_size,
    problems,
    temp,
    is_reasoning,
    prompt_template=None,
    max_tokens=31_000,
):
    print(f"\n--- Loading {model_path} (TP={tp_size}) ---")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
    )

    tokenizer = llm.get_tokenizer()
    inputs = []
    
    for p in problems:
        

        if not is_reasoning:
            # DIRECT MODE: We will force the start of the assistant message
            user_content = f"Problem: {p}\n\nOutput ONLY the final answer inside \\boxed{{}}."
            messages = [{"role": "user", "content": user_content}]
            
            # Apply template but DO NOT generate the prompt yet
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=is_reasoning,
            )
            
            # 2. Manually append the tokens for "\boxed{"
            prompt_text = tokenizer.decode(prompt_ids) + "\\boxed{"
            inputs.append(prompt_text)
            
        else:
            # COT MODE
            messages = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": p}
            ]
            inputs.append(tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True
            ))

    # 3. Adjust Stop Tokens
    # If we forced "\boxed{", we need to accept that the model might just output "5}" and stop.
    limit = max_tokens if is_reasoning else 50 # Very short limit for direct answers
    
    params = SamplingParams(temperature=temp, max_tokens=limit)
    outputs = llm.generate(inputs, params)

    generated_texts = []
    for o in outputs:
        text = o.outputs[0].text
        
        # 4. Reconstruct the full answer for the Verifier
        # If we pre-filled "\boxed{", the model only gave us the content inside/after.
        if not is_reasoning:
            # We add the box back so the Regex finds it
            text = "\\boxed{" + text 
            
        generated_texts.append(text)

        print("Pred:", text)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--- Unloaded {model_path} ---")

    return generated_texts

# MAIN FILTERING LOGIC

def filter_dataset():
    # Load initial data
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(ds)
    print(f"Initial pool: {len(df)} problems")

    # PHASE 1: JUNIOR MODEL
    # M_jun must be INCORRECT with Direct and CoT

    # Step 1.1: Junior Direct
    # If junior gets it right directly, it's too easy. Discard.
    print("\nPhase 1.1: Checking Junior Direct (Expect Failure)")
    jun_direct_outputs = run_batch_inference(
        MODEL_JUNIOR,
        TENSOR_PARALLEL_SIZE_JUNIOR,
        df['problem'].tolist(),
        temp=0.0,
        is_reasoning=False,
        prompt_template=None,
    )

    keep_indices = []
    for idx, (gen, truth) in enumerate(zip(jun_direct_outputs, df['solution'])):
        if not MathVerifier.is_correct(gen, truth):
            keep_indices.append(idx)

    df = df.iloc[keep_indices].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Junior Failed Direct)")
    if len(df) == 0: return

    # Step 1.2: Junior CoT
    print("\nPhase 1.2: Checking Junior CoT (Expect Failure)")
    jun_cot_outputs = run_batch_inference(
        MODEL_JUNIOR,
        TENSOR_PARALLEL_SIZE_JUNIOR,
        df['problem'].tolist(),
        temp=0.0,
        is_reasoning=True,
        prompt_template=PROMPT_COT,
    )

    keep_indices = []
    for idx, (gen, truth) in enumerate(zip(jun_cot_outputs, df['solution'])):
        # KEEP if incorrect
        if not MathVerifier.is_correct(gen, truth):
            keep_indices.append(idx)

    df = df.iloc[keep_indices].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Junior Failed CoT)")
    if len(df) == 0: return

    # PHASE 2: SENIOR MODEL (M_sen)
    # Goal: Keep problems where M_sen FAILS Direct but SUCCEEDS with CoT

    # 2.1 Senior Direct
    print("\nPhase 2.1: Checking Senior Direct (Expect Failure)")
    sen_direct_outputs = run_batch_inference(
        MODEL_SENIOR,
        TENSOR_PARALLEL_SIZE_SENIOR,
        df['problem'].tolist(),
        temp=0.0,
        is_reasoning=False,
    )

    keep_indices = []
    for idx, (gen, truth) in enumerate(zip(sen_direct_outputs, df['solution'])):
        # KEEP if Incorrect
        if not MathVerifier.is_correct(gen, truth):
            keep_indices.append(idx)

    df = df.iloc[keep_indices].reset_index(drop=True)
    print(f"-> Kept {len(df)} problems (Senior Failed Direct)")
    if len(df) == 0: return

    # 2.2 Senior CoT
    print("\n[Phase 2.2] Checking Senior CoT (Expect Success)...")
    sen_cot_outputs = run_batch_inference(
        MODEL_SENIOR, TENSOR_PARALLEL_SIZE_SENIOR,
        df['problem'].tolist(),
        temp=0.0,
        is_reasoning=True,
        prompt_template=PROMPT_COT
    )

    keep_indices = []
    valid_cots = []
    
    for idx, (gen, truth) in enumerate(zip(sen_cot_outputs, df['solution'])):
        # KEEP if Correct
        if MathVerifier.is_correct(gen, truth):
            keep_indices.append(idx)
            valid_cots.append(gen) # Save the CoT!

    df = df.iloc[keep_indices].reset_index(drop=True)
    df['senior_cot'] = valid_cots
    
    print(f"-> Final Dataset Size: {len(df)} problems")

    # SAVE
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_json(OUTPUT_FILE, orient='records', lines=True)
    print("Done.")

if __name__ == "__main__":
    filter_dataset()