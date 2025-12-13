import os
import re
import json
import torch
import random
import numpy as np
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---

# Infrastructure
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
TENSOR_PARALLEL_SIZE = 2 # 7B might fit on 1 GPU, but 2 is safe/fast given your setup
SEED = 9001

MODEL_NAME = "Qwen/Qwen3-8B"

INPUT_FILE = "experiment_1_results.jsonl" 
OUTPUT_FILE = "experiment_2_hsp_results.jsonl"

# Parameters
N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class MathVerifier:
    @staticmethod
    def extract_answer(text):
        if not text: return None
        # Handle cases where whitespace is inserted like \boxed { content }
        start_indices = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
        if not start_indices: return None
        
        for start in start_indices:
            balance = 0
            content_start = text.find('{', start) + 1
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{': balance += 1
                elif char == '}':
                    if balance == 0: return text[content_start:i]
                    balance -= 1
        return None

    @staticmethod
    def normalize_answer(text):
        if text is None: return ""
        text = text.strip().replace(" ", "")
        
        # --- FIXES FROM EXP 1 ---
        text = text.replace("\\\\", "\\") # Fix JSON escaping
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        text = text.replace(r"\left", "").replace(r"\right", "") # Remove sizing
        text = re.sub(r'\\text\{(.*?)\}', r'\1', text) # Remove text wrappers
        
        return text

    @staticmethod
    def is_correct(generated_text, ground_truth):
        # We only need standard extraction here because we let Junior finish naturally
        pred = MathVerifier.extract_answer(generated_text)
        truth = MathVerifier.extract_answer(ground_truth)
        
        if truth is None: truth = ground_truth
        
        return MathVerifier.normalize_answer(pred) == MathVerifier.normalize_answer(truth)
    
class HSPExperiment:
    def __init__(self):
        print(f"Loading Junior Model: {MODEL_NAME}...")
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True, 
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.90
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        # Sampling for Handoff: Allow full CoT generation
        self.params_handoff = SamplingParams(
            n=N_SAMPLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED
        )

    def run_handoff(self, datum):
        problem = datum['problem']
        ground_truth = datum['solution']
        steps = datum['steps'] # Reuse steps from Exp 1 for alignment
        
        print(f"\nProblem ID: {datum.get('id', 'N/A')} | Steps: {len(steps)}")
        
        # 1. Prepare Prompt Prefix
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Iterate k (Handoff Points)
        hsp_index = -1
        step_accuracies = []
        
        # Optimization: Only run until we find stable HSP to save compute?
        # The user requested seeing if HSP < ESSP, so we should start from k=0.
        
        for k in range(len(steps)):
            partial_reasoning = " ".join(steps[:k+1])
            
            # --- OPTIMIZATION: Answer already revealed? ---
            # If the Senior model already boxed the answer in this step,
            # the Junior model (Auditor) can see it immediately.
            if "\\boxed" in steps[k]:
                print(f"  Step {k}: Answer visible in prefix. Auto-Success.")
                step_accuracies.append(1.0)
                if hsp_index == -1: hsp_index = k
                # We can skip generation here, or continue if you want full curves
                continue 

            # --- HANDOFF GENERATION ---
            # Prompt: Base + <think> + Partial Steps (NO Trigger)
            # We let Junior resume thinking from where Senior left off.
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
                if hsp_index == -1:
                    hsp_index = k
                    print(f"  [HSP FOUND] at k={k}")
                    # If you want to save compute, you can break here.
                    # Given the research pivot, getting the full curve might be useful
                    # but breaking is standard for 'finding the point'.
                    break 

        if hsp_index == -1:
            print("  [FAILURE] Junior never recovered the solution.")

        return {
            "id": datum.get('id'),
            "problem": problem,
            "solution": ground_truth,
            "total_steps": len(steps),
            "essp_index": datum.get('essp_index'), # Carry over ESSP for easy comparison
            "hsp_index": hsp_index,
            "hsp_step_accuracies": step_accuracies
        }

    def run_experiment(self):
        print("Loading Exp 1 Results...")
        data = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: pass
        
        print(f"Loaded {len(data)} problems.")
        
        # Open output file
        with open(OUTPUT_FILE, 'w') as f_out:
            for i, datum in enumerate(data):
                # Only process problems that Senior actually solved (ESSP != -1)
                # If Senior failed, Handoff is undefined/moot.
                if datum.get('essp_index') == -1:
                    print(f"Skipping ID {datum.get('id')}: Senior failed (ESSP=-1).")
                    continue
                    
                result = self.run_handoff(datum)
                
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    experiment = HSPExperiment()
    experiment.run_experiment()
