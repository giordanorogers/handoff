import os
import re
import json
import torch
import random
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.utils import MathVerifier, CoTSplitter

SEED = 9001

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
TENSOR_PARALLEL_SIZE = 2

MODEL_NAME = "Qwen/Qwen3-32B"

INPUT_FILE = "data/filtered_math_dataset.jsonl"
OUTPUT_FILE = "data/experiment_1_results.jsonl"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5

TRIGGER_PHRASE = "\n</think>\nTherefore, the final answer is \\boxed{"

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class ESSPExperiment:
    """Early stopping success point experiment."""
    
    def __init__(self):
        
        # Initialize vLLM
        print(f"Loading {MODEL_NAME}...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True,
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # Forced answer sampling
        # Short max tokens because we force the answer immediately
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
        
        if match :
            return match.group(1).strip()
        
        # Fallback: try to remove the \boxed{} part at the end
        print("WARNING: Couldn't extract text from <think> tags. Falling back to raw_cot.")
        if "\\boxed" in raw_cot:
            return raw_cot.split("\\boxed")[0].strip()
        return raw_cot
    
    def run_problem(self, data):
        problem = data['problem']
        ground_truth = data['solution']
        full_cot = data['senior_cot']
        
        # 1. Prepare Steps
        reasoning_body = self.extract_reasoning_body(full_cot)
        steps = CoTSplitter.split(reasoning_body)
        
        if not steps:
            print(f"WARNING: Couldn't extract steps. Returning None.")
            return None
        
        # 2. Pre-calculate prompt prefix (System + User)
        # We use the prompt template for the static part
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
        result_data = {
            "problem": problem,
            "solution": ground_truth,
            "full_cot": full_cot,
            "steps": steps,
            "total_steps": len(steps),
            "first_essp_index": -1,     # Default failure
            "essp_indices": [],
            "step_accuracies": []
        }
        
        for k in range(len(steps)):
            # Construct Partial CoT
            # We are manually constructing the assistant's turn inside <think>
            partial_reasoning = " ".join(steps[:k+1])
            
            used_trigger = False
            
            if "\\boxed" in steps[k]:
                prompt = f"{base_prompt}<think>\n{partial_reasoning}"
            else:
                # Construct Prompt: Base + <think> + Partial + Trigger
                prompt = f"{base_prompt}<think>\n{partial_reasoning}{TRIGGER_PHRASE}"
                used_trigger = True
                
            # Generate
            # vLLM handles batching internally
            outputs = self.llm.generate([prompt], self.params_probe, use_tqdm=False)
            
            # Process Outputs
            completions = [o.text for o in outputs[0].outputs]
            
            # Verify
            correct_count = 0
            for comp in completions:
                # We forced the box open, so we add it back for the verifier
                if used_trigger:
                    # Pass raw completion (e.g., "7/20")
                    if MathVerifier.is_correct(comp, ground_truth, is_partial=True):
                        correct_count += 1
                else:
                    # Pass raw completion (e.g., "Therefore \boxed{7/20}")
                    if MathVerifier.is_correct(comp, ground_truth, is_partial=False):
                        correct_count += 1
            
            accuracy = correct_count / N_SAMPLES
            result_data['step_accuracies'].append(accuracy)
            
            print(f"    Step {k}/{len(steps)-1}: Acc={accuracy:.2f} | '{steps[k][:30]}...")
            
            # Check Threshold
            if accuracy >= THRESHOLD_TAU:
                if result_data['first_essp_index'] == -1:
                    result_data['first_essp_index'] = k
                result_data['essp_indices'].append(k)
                print(f"    [ESSP FOUND] at k={k}")
                essp_found = True
                
        if not essp_found:
            print(" [FAILURE] Reached end of CoT without hitting threshold.")
            
        return result_data
    
    def run_experiment(self):
        # Load filtered dataset
        dataset = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
                
        print(f"Loaded {len(dataset)} problems from {INPUT_FILE}")
        
        # Open output file for appending (safe against crashes)
        with open(OUTPUT_FILE, 'w') as f_out:
            for i, data in enumerate(dataset):
                # Add ID if missing
                if 'id' not in data:
                    data['id'] = i
                
                result = self.run_problem(data)
                
                if result:
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush() # Ensure write to disk
        
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    experiment = ESSPExperiment()
    experiment.run_experiment()
