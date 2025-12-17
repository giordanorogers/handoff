import os
import re
import json
import torch
import random
import numpy as np
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
TENSOR_PARALLEL_SIZE = 1
SEED = 9001

MODEL_NAME = "Qwen/Qwen3-4B"

INPUT_FILE = "data/MATH_92/MATH_92.jsonl" 
OUTPUT_FILE = "data/experiment_2_hsp_results.jsonl"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class HSPExperiment:
    """Handoff success point experiment."""
    
    def __init__(self):
        
        print(f"Loading Junior Model: {MODEL_NAME}...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True,
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # Sampling for handoff: Allow full CoT generation
        self.params_handoff = SamplingParams(
            n=N_SAMPLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED
        )
        
    def run_handoff(self, data):
        problem = data['problem']
        ground_truth = data['solution']
        steps = data['steps']
        
        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {len(steps)}")
        
        # 1. Prepare prompt prefix
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 1.1 Verify Junior cannot get the correct answer on its own.
        outputs = self.llm.generate([base_prompt], self.params_handoff, use_tqdm=True)
        completions = [o.text for o in outputs[0].outputs]
        
        # Verify
        correct_count = 0
        for comp in completions:
            if MathVerifier.is_correct(comp, ground_truth):
                correct_count += 1
                
        accuracy = correct_count / N_SAMPLES
        
        print(f"  Base Acc={accuracy:.2f}")
        
        if accuracy > 0.0:
            return {
                "id": data.get('id'),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": 0,
                "essp_index": data.get('first_essp_index'),
                "first_hsp_index": -1,
                "hsp_indices": [],
                "hsp_step_accuracies": []
            }
        
        # 2. Iterate k (handoff points)
        first_hsp_index = -1
        hsp_indices = []
        step_accuracies = []
        
        for k in range(len(steps)):
            partial_reasoning = " ".join(steps[:k+1])
            
            # If the Senior model already boxed the answer in this step,
            # the Junior model can see it immediately.
            if "\\boxed" in steps[k]:
                print(f"  Step {k}: Answer visible in prefix. Auto-Success.")
                step_accuracies.append(1.0)
                if first_hsp_index == -1:
                    first_hsp_index = k
            
            # Handoff Generation
            # Prompt: Base + <think> + Partial Steps
            # We let the Junior resume thinking from where the Senior left off.
            prompt = f"{base_prompt}<think>\n{partial_reasoning}"
            
            # Generate
            outputs = self.llm.generate([prompt], self.params_handoff, use_tqdm=True)
            completions = [o.text for o in outputs[0].outputs]
            
            # Verify
            correct_count = 0
            for comp in completions:
                # We combine prefix + completion to extract the answer
                # (in case Junior closes the box immediately)
                full_text = partial_reasoning + comp
                if MathVerifier.is_correct(full_text, ground_truth):
                    correct_count += 1
                    
            accuracy = correct_count / N_SAMPLES
            step_accuracies.append(accuracy)
            
            print(f"  Step {k}/{len(steps)-1}: Acc={accuracy:.2f} | Last: '{steps[k][-30:] if len(steps[k])>30 else steps[k]}'")
            
            # Check HSP
            if accuracy >= THRESHOLD_TAU:
                if first_hsp_index == -1:
                    first_hsp_index = k
                print(f"  [HSP FOUND] at k={k}")
                hsp_indices.append(k)
                    
        if first_hsp_index == -1:
            print("  [FAILURE] Junior never recovered the solution.")
            
        return {
            "id": data.get('id'),
            "problem": problem,
            "solution": ground_truth,
            "total_steps": len(steps),
            "first_essp_index": data.get('first_essp_index'),
            "essp_indices": data.get('essp_indices'),
            "first_hsp_index": first_hsp_index,
            "hsp_indices": hsp_indices,
            "hsp_step_accuracies": step_accuracies
        }
        
    def run_experiment(self):
        print("Loading ESSP Results...")
        dataset = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass
                
        print(f"Loaded {len(dataset)} problems.")
        
        # Open output file
        with open(OUTPUT_FILE, 'w') as f_out:
            for i, data in enumerate(dataset):
                # Only process probelms that the Senior actually solved (ESSP != -1)
                # If Senior failed, Handoff is undefined/moot.
                if data.get('first_essp_index') == -1:
                    print(f"Skipping ID {data.get('id')}: Senior failed (ESSP=-1).")
                    continue
                
                result = self.run_handoff(data)
                
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    experiment = HSPExperiment()
    experiment.run_experiment()
