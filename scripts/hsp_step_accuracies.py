import os
import re
import json
import torch
import random
import argparse
import numpy as np
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

TENSOR_PARALLEL_SIZE = 1
SEED = 9001

MODEL_NAME = "Qwen/Qwen3-4B"

INPUT_FILE = "data/exp_2_hsp_res.jsonl"
OUTPUT_FILE = "data/hsp_step_accuracies.jsonl"

N_SAMPLES = 5
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

CHECKPOINT_CHUNK_SIZE = 50

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

class HSSPStepAccuracies:
    """
    Handoff success step accuracies experiment.
    Finds the final answer prediction accuracy at each handed off step.
    """
    
    def __init__(self, gpu_id=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Loading Junior Model: {MODEL_NAME} on GPU {gpu_id}...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True,
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.params_handoff = SamplingParams(
            n=N_SAMPLES,
            temperature=0.7,
            top_p=0.95,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED,
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

    def run_handoff(self, data, output_file):
        """Find step accuracies with chunked checkpointing."""
        problem = data['problem']
        ground_truth = data['solution']
        steps = data['steps']
        problem_id = data.get('id')
        
        print(f"\nProblem ID: {problem_id} | Steps: {len(steps)}")
        
        # Load existing progress
        hsp_step_accuracies = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record['id'] == problem_id:
                            hsp_step_accuracies = record.get('hsp_step_accuracies', [])
                            print(f"  Resuming from step {len(hsp_step_accuracies)}")
                            break
                    except:
                        pass
        
        start_step = len(hsp_step_accuracies)
        if start_step >= len(steps):
            print(f"  Already complete.")
            return None
        
        base_prompt = self._build_base_prompt(problem)
        
        # Process in chunks
        for chunk_start in range(start_step, len(steps), CHECKPOINT_CHUNK_SIZE):
            chunk_end = min(chunk_start + CHECKPOINT_CHUNK_SIZE, len(steps))
            
            # Prepare prompts for this chunk
            prompt_data = []
            for k in range(chunk_start, chunk_end):
                partial_reasoning = " ".join(steps[:k+1])
                has_boxed = "\\boxed" in steps[k]
                prompt = f"{base_prompt}<think>\n{partial_reasoning}"
                prompt_data.append((k, prompt, partial_reasoning, has_boxed))
            
            steps_needing_generation = [
                (k, prompt, partial, has_boxed)
                for k, prompt, partial, has_boxed in prompt_data
                if not has_boxed
            ]
            
            # Batched generation for this chunk
            generation_results = {}
            if steps_needing_generation:
                prompts_to_generate = [item[1] for item in steps_needing_generation]
                print(f"  Chunk [{chunk_start}-{chunk_end}]: generating {len(prompts_to_generate)} prompts...")
                
                outputs = self.llm.generate(
                    prompts_to_generate, self.params_handoff, use_tqdm=True
                )
                
                for idx, (k, prompt, partial_reasoning, _) in enumerate(steps_needing_generation):
                    generation_results[k] = (outputs[idx], partial_reasoning)
            
            # Compute accuracies for this chunk
            for k, prompt, partial_reasoning, has_boxed in prompt_data:
                if has_boxed:
                    accuracy = 1.0
                else:
                    output, partial = generation_results[k]
                    completions = [o.text for o in output.outputs]
                    correct_count = sum(
                        1 for comp in completions
                        if MathVerifier.is_correct(partial + comp, ground_truth)
                    )
                    accuracy = correct_count / N_SAMPLES
                
                print(f"  Step {k}/{len(steps)-1}: Acc={accuracy:.2f}")
                hsp_step_accuracies.append(accuracy)
            
            # Checkpoint after each chunk
            self._save_checkpoint(output_file, data, hsp_step_accuracies)
            print(f"  [CHECKPOINTED] through step {chunk_end - 1}")
        
        return self._build_result(data, hsp_step_accuracies)

    def _save_checkpoint(self, output_file, data, hsp_step_accuracies):
        """Save/update checkpoint for a problem."""
        problem_id = data.get('id')
        
        records = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        records[record['id']] = record
                    except:
                        pass
        
        records[problem_id] = self._build_result(data, hsp_step_accuracies)
        
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            for record in records.values():
                f.write(json.dumps(record) + "\n")
        os.replace(temp_file, output_file)  # atomic on POSIX

    def _build_result(self, data, hsp_step_accuracies):
        return {
            "id": data.get("id"),
            "problem": data['problem'],
            "solution": data['solution'],
            "full_cot": data.get("full_cot"),
            "steps": data['steps'],
            "total_steps": data.get("total_steps"),
            "first_essp_index": data.get("first_essp_index"),
            "essp_step_accuracies": data.get("step_accuracies"),
            "essp_indices": data.get("essp_indices"),
            "answer": data.get("answer"),
            "first_hsp_index": data.get("first_hsp_index"),
            "hsp_step_accuracies": hsp_step_accuracies
        }

    def run_experiment(self, shard=0, num_shards=1):
        print("Loading Data...")
        dataset = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass
        
        if num_shards > 1:
            output_file = OUTPUT_FILE.replace(".jsonl", f"_shard{shard}.jsonl")
        else:
            output_file = OUTPUT_FILE
        
        dataset = [p for i, p in enumerate(dataset) if i % num_shards == shard]
        print(f"{len(dataset)} problems for shard {shard}/{num_shards}.")
        
        for data in dataset:
            self.run_handoff(data, output_file)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment = HSSPStepAccuracies(gpu_id=args.gpu)
    experiment.run_experiment(shard=args.shard, num_shards=args.num_shards)