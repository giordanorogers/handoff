import time
import os
import re
import json
import torch
import random
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
TENSOR_PARALLEL_SIZE = 2
SEED = 9001

MODEL_NAME = "Qwen/Qwen3-32B"

INPUT_FILE = "filtered_math_dataset.jsonl"
OUTPUT_FILE = "experiment_1_results.jsonl"
N_SAMPLES = 20
THRESHOLD_TAU = 0.5

TRIGGER_PHRASE = "\n</think>\nTherefore, the final answer is \\boxed{"

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class MathVerifier:
    """Robust Answer Extractor with Normalization fixes"""
    
    @staticmethod
    def extract_answer(text):
        if not text: return None
        # Handle cases where whitespace is inserted like \boxed { content }
        start_indices = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
        if not start_indices: return None
        
        for start in start_indices:
            balance = 0
            # Find the first '{' after the \boxed
            content_start = text.find('{', start) + 1
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{': balance += 1
                elif char == '}':
                    if balance == 0: return text[content_start:i]
                    balance -= 1
        return None

    @staticmethod
    def extract_from_partial(text):
        """
        Extracts content when the prompt ends with \boxed{
        We look for the first closing brace '}' that isn't balanced by an opening '{'
        within the generated text itself.
        """
        if not text: return None
        balance = 0
        for i, char in enumerate(text):
            if char == '{':
                balance += 1
            elif char == '}':
                # If balance is 0, this '}' closes the ghost \boxed{ from the prompt
                if balance == 0:
                    return text[:i]
                balance -= 1
        # Fallback: if no closing brace found, return everything (model stopped early)
        return text

    @staticmethod
    def normalize_answer(text):
        if text is None: return ""
        text = text.strip().replace(" ", "")
        
        # --- FIX 1: Handle JSON-style double escaping ---
        # Turns \\left into \left, \\dfrac into \dfrac
        text = text.replace("\\\\", "\\") 
        
        # --- FIX 2: Standardize LaTeX fractions ---
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        
        # --- FIX 3: Remove sizing commands ---
        # \left( and \right) are visual formatting; removing them makes 
        # comparisons robust against stylistic differences.
        text = text.replace(r"\left", "").replace(r"\right", "")
        
        # Fix 4: Remove \text{...} wrappers
        text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
        
        return text

    @staticmethod
    def is_correct(generated_text, ground_truth, is_partial=False):
        if is_partial:
            pred = MathVerifier.extract_from_partial(generated_text)
        else:
            pred = MathVerifier.extract_answer(generated_text)
            
        truth = MathVerifier.extract_answer(ground_truth)
        if truth is None: truth = ground_truth
        
        # Debug print to verify the fix works in your logs
        print()
        print(f"  Norm Pred: '{MathVerifier.normalize_answer(pred)}'")
        print(f"  Norm Truth: '{MathVerifier.normalize_answer(truth)}'")

        return MathVerifier.normalize_answer(pred) == MathVerifier.normalize_answer(truth)

class CoTSplitter:
    """
    Splits reasoning text into logical steps (sentences), 
    but protects LaTeX math environments from being split.
    """
    @staticmethod
    def split(text):
        # 1. Identify chunks: Math vs Text
        # Regex to find LaTeX patterns: $...$, $$...$$, \[...\], \(...\)
        # We perform a split that keeps the delimiters
        pattern = r'(\$\$[\s\S]*?\$\$|\$[\s\S]*?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))'
        chunks = re.split(pattern, text)
        
        steps = []
        current_step = ""
        
        # 2. Iterate chunks
        for chunk in chunks:
            # If this chunk is a LaTeX block, treat it as a single indivisible atom
            if re.match(pattern, chunk):
                current_step += chunk
                continue
            
            # If it's text, we can split by sentence delimiters
            # We look for punctuation followed by whitespace or end of string
            # We avoid splitting abbreviations (simplified check)
            sub_parts = re.split(r'([.?!]\s+)', chunk)
            
            for part in sub_parts:
                current_step += part
                # If this part ends with a delimiter, it might be a split point
                if re.match(r'[.?!]\s+', part):
                    # Check if the current step is long enough to be a sentence
                    if len(current_step.strip()) > 5:
                        steps.append(current_step.strip())
                        current_step = ""
        
        if current_step.strip():
            steps.append(current_step.strip())
            
        return steps
    
class ESSPExperiment:
    def __init__(self):
        # Initialize vLLM
        print(f"Loading {MODEL_NAME}...")
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True,
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.90
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Sampling for the N=20 probe
        # Short max_tokens because we force the answer immediately
        self.params_probe = SamplingParams(
            n=N_SAMPLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=25,
            seed=SEED
        )

    def extract_reasoning_body(self, raw_cot):
        """
        Extracts text inside <think>...</think>
        If no tags, returns the whole text (fallback).
        """
        match = re.search(r"<think>(.*?)</think>", raw_cot, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: try to remove the \boxed{} part at the end
        if "\\boxed" in raw_cot:
            return raw_cot.split("\\boxed")[0].strip()
        return raw_cot
    
    def run_problem(self, datum):
        problem = datum['problem']
        ground_truth = datum['solution']
        full_cot = datum['senior_cot'] # From out filter script

        # 1. Prepare Steps
        reasoning_body = self.extract_reasoning_body(full_cot)
        steps = CoTSplitter.split(reasoning_body)

        if not steps:
            return None
        
        print(f"\nProblem ID: {datum.get('id', 'N/A')} | Total Steps: {len(steps)}")

        # 2. Pre-calculate prompt prefix (System + User)
        # We use the chat template for the static part
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        # We get the formatted text up to the start of the assistant
        base_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. Iterate k
        essp_found = False
        result_data = {
            "problem": problem,
            "solution": ground_truth,
            "full_cot": full_cot,
            "steps": steps,
            "total_steps": len(steps),
            "essp_index": -1,           # Default failure
            "step_accuracies": []       # For the plot
        }
        

        for k in range(len(steps)):
            # Construct Partial CoT
            # We are manually constructing the Assistant's turn inside <think>
            partial_reasoning = " ".join(steps[:k+1])

            used_trigger = False

            if "\\boxed" in steps[k]:
                prompt = f"{base_prompt}<think>\n{partial_reasoning}"
            else:
                # Construct Prompt: Base + <think> + Partial + Trigger
                prompt = f"{base_prompt}<think>\n{partial_reasoning}{TRIGGER_PHRASE}"
                used_trigger=True

            # Generate
            # cLLM handles the n=20 batching internally
            outputs = self.llm.generate([prompt], self.params_probe, use_tqdm=False)

            # Process Outputs
            completions = [o.text for o in outputs[0].outputs]

            # Verify
            correct_count = 0
            for comp in completions:
                # We forced the box open, so we add it back for the verifier
                if used_trigger:
                    # Pass raw completion (e.g., "7/20}")
                    if MathVerifier.is_correct(comp, ground_truth, is_partial=True):
                        correct_count += 1
                else:
                    # Pass raw completion (e.g., "Therefore \boxed{7/20}")
                    if MathVerifier.is_correct(comp, ground_truth, is_partial=False):
                        correct_count += 1

            accuracy = correct_count / N_SAMPLES
            result_data["step_accuracies"].append(accuracy)

            print(f"    Step {k}/{len(steps)-1}: Acc={accuracy:.2f} | '{steps[k][:30]}...")

            # Check Threshold
            if accuracy >= THRESHOLD_TAU:
                result_data["essp_index"] = k
                print(f"    [ESSP FOUND] at k={k}")
                essp_found = True
                break # Stop search

        if not essp_found:
            print(" [FAILURE] Reached end of CoT without hitting threshold.")

        return result_data

    def run_experiment(self):
        # Load filtered data
        data = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        print(f"Loaded {len(data)} problems from {INPUT_FILE}")

        # Open output file for appending (safe against crashes)
        with open(OUTPUT_FILE, 'w') as f_out:
            for i, datum in enumerate(data):
                # Optional: Add ID if missing
                if 'id' not in datum: datum['id'] = i
                
                result = self.run_problem(datum)
                
                if result:
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush() # Ensure write to disk

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment = ESSPExperiment()
    experiment.run_experiment()
