import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

INPUT_FILE = "data_/problem2_essp.jsonl"
OUTPUT_FILE = "data_/problem2_hsp.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class HSPExperimentFirstOnly:
    """Handoff success point experiment"""
    
    def __init__(
        self,
        gpu_id=0,
        model_name="Qwen/Qwen3-4B",
        temperature=0.6,
        n_samples=50,
        seed=9001,
        top_p=0.9,
        max_tokens=31_000
    ):
        self.n_samples = n_samples
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Loading Junior Model: {model_name} on GPU {gpu_id}...")
        
        self.llm = LLM(
            model=model_name,
            seed=seed,            
            enable_prefix_caching=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.params_handoff = SamplingParams(
            n=self.n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        

    def _build_base_prompt(self, problem):
        """Build the base prompt for a problem."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def run_handoff(self, data):
        """Find first HSP using batched generation with early stopping."""
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]
        
        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {len(steps)}")
        
        base_prompt = self._build_base_prompt(problem)
            
        step_accuracies = []
        first_hsp_index = None  # Initialize to None
        
        for i in range(len(steps)):
            partial_reasoning = "\n".join(steps[:i+1])
            prompt = f"{base_prompt}<think>\n{partial_reasoning}"
            
            outputs = self.llm.generate([prompt], self.params_handoff, use_tqdm=False)
            completions = [o.text for o in outputs[0].outputs]
            
            correct_count = sum(
                1 for comp in completions if MathVerifier.is_correct(comp, ground_truth)
            )
            step_accuracy = correct_count / self.n_samples
            step_accuracies.append(step_accuracy)
            
            if step_accuracy > 0.5:
                first_hsp_index = i
                break
        
        return {
            "id": data.get("id"),
            "problem": problem,
            "solution": ground_truth,
            "answer": data.get("answer"),
            "total_steps": len(steps),
            "steps": steps,
            "first_essp_index": data.get("first_essp_index"),
            "essp_indices": data.get("essp_indices"),
            "essp_step_accuracies": data.get("step_accuracies"),
            "first_hsp_index": first_hsp_index,  # Will be None if no HSP found
            "hsp_step_accuracies": step_accuracies,
        }
            
    def run_experiment(self):
        """Run the experiment on a shard of the data."""
        print("Loading ESSP results...")
        dataset = []
        with open(INPUT_FILE, "r") as f_in:
            for line in f_in:
                dataset.append(json.loads(line))
                
        print(dataset[0].keys())
                
        with open(OUTPUT_FILE, "w") as f_out:
            for i, data in enumerate(dataset):
                result = self.run_handoff(data)
                
                f_out.write(json.dumps(result) + '\n')
                f_out.flush()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9001, help="Random seed")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="LM name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiment = HSPExperimentFirstOnly(
        gpu_id=args.gpu,
        model_name=args.model_name,
        seed=args.seed           
    )
    experiment.run_experiment()