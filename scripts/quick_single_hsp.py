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

INPUT_FILE = "data/MATH_92/MATH_92_focused.jsonl"
CHECKPOINT_FILE = "data/MATH_92/MATH_92_hsp_focused.jsonl"
OUTPUT_FILE = "data/experiment_2_hsp_results_focused.jsonl"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


class HSPExperimentSequential:
    """Handoff success point experiment - sequential with true early stopping."""

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

    def _check_accuracy_at_step(self, base_prompt, steps, k, ground_truth):
        """Check handoff accuracy at step k. Returns accuracy float."""
        partial_reasoning = " ".join(steps[:k + 1])
        
        # If correct boxed answer already in partial, no generation needed
        if "\\boxed" in partial_reasoning:
            if MathVerifier.is_correct(partial_reasoning, ground_truth):
                return 1.0
        
        prompt = f"{base_prompt}<think>\n{partial_reasoning}"
        outputs = self.llm.generate([prompt], self.params_handoff, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]
        
        correct_count = sum(
            1 for comp in completions 
            if MathVerifier.is_correct(partial_reasoning + comp, ground_truth)
        )
        return correct_count / N_SAMPLES

    def run_handoff(self, data):
        """Find first HSP using sequential generation with true early stopping."""
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]

        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {len(steps)}")

        base_prompt = self._build_base_prompt(problem)

        # Verify Junior cannot solve on its own
        base_accuracy = self._check_accuracy_at_step(base_prompt, [], -1, ground_truth)
        # Actually, for baseline we need a different check - no partial reasoning
        prompt_baseline = base_prompt
        outputs = self.llm.generate([prompt_baseline], self.params_handoff, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]
        base_accuracy = sum(
            1 for comp in completions 
            if MathVerifier.is_correct(comp, ground_truth)
        ) / N_SAMPLES
        
        print(f"  Base Acc={base_accuracy:.2f}")

        if base_accuracy > 0.0:
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": len(steps),
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": -1,
                "skipped_reason": "junior_baseline_nonzero",
            }

        # Sequential scan with true early stopping
        first_hsp_index = -1
        
        for k in range(len(steps)):
            accuracy = self._check_accuracy_at_step(base_prompt, steps, k, ground_truth)
            print(f"  Step {k}/{len(steps)-1}: Acc={accuracy:.2f}")
            
            if accuracy >= THRESHOLD_TAU:
                first_hsp_index = k
                print(f"  [FIRST HSP FOUND] at k={k}")
                break  # TRUE early stopping - no more generation

        if first_hsp_index == -1:
            print("  [FAILURE] Junior never recovered the solution.")

        return {
            "id": data.get("id"),
            "problem": problem,
            "solution": ground_truth,
            "total_steps": len(steps),
            "first_essp_index": data.get("first_essp_index"),
            "essp_indices": data.get("essp_indices"),
            "first_hsp_index": first_hsp_index,
        }

    def run_experiment(self, shard=0, num_shards=1):
        """Run the experiment on a shard of the data."""
        print("Loading ESSP Results...")
        dataset = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass
                
        checkpoint_ids = []
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    checkpoint_ids.append(data['id'])
                
        # Filter to problems not in the checkpoint file
        dataset = [p for p in dataset if p['id'] not in checkpoint_ids]

        # Filter to this shard
        dataset = [p for i, p in enumerate(dataset) if i % num_shards == shard]
        print(f"Loaded {len(dataset)} problems for shard {shard}/{num_shards}.")

        # Output file per shard
        if num_shards > 1:
            output_file = OUTPUT_FILE.replace(".jsonl", f"_shard{shard}.jsonl")
        else:
            output_file = OUTPUT_FILE

        with open(output_file, "a") as f_out:  # Changed to append mode
            for i, data in enumerate(dataset):
                if data.get("first_essp_index") == -1:
                    print(f"Skipping ID {data.get('id')}: Senior failed (ESSP=-1).")
                    continue

                result = self.run_handoff(data)

                f_out.write(json.dumps(result) + "\n")
                f_out.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID to use")
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment = HSPExperimentSequential(gpu_id=args.gpu)
    experiment.run_experiment(shard=args.shard, num_shards=args.num_shards)