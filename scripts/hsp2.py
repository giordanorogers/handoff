import os
import re
import json
import torch
import random
import numpy as np
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TENSOR_PARALLEL_SIZE = 1
SEED = 9001

MODEL_NAME = "Qwen/Qwen3-4B"

INPUT_FILE = "data/MATH_92/MATH_92.jsonl"
OUTPUT_FILE = "data/experiment_2.1_hsp_results.jsonl"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


class HSPExperiment:
    """Handoff success point experiment (batched version)."""

    def __init__(self):
        print(f"Loading Junior Model: {MODEL_NAME}...")

        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_prefix_caching=True,
            seed=SEED,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Sampling for handoff: Allow full CoT generation
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

    def _check_base_accuracy(self, base_prompt, ground_truth):
        """Check if Junior can solve the problem without any hints."""
        outputs = self.llm.generate([base_prompt], self.params_handoff, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]

        correct_count = sum(
            1 for comp in completions if MathVerifier.is_correct(comp, ground_truth)
        )
        return correct_count / N_SAMPLES

    def run_handoff(self, data):
        """Run handoff experiment for a single problem with batched generation."""
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]

        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {len(steps)}")

        # 1. Build base prompt
        base_prompt = self._build_base_prompt(problem)

        # 2. Verify Junior cannot get the correct answer on its own
        base_accuracy = self._check_base_accuracy(base_prompt, ground_truth)
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
                "hsp_indices": [],
                "hsp_step_accuracies": [],
                "skipped_reason": "junior_baseline_nonzero",
            }

        # 3. Prepare ALL prompts for batched generation
        prompt_data = []  # List of (step_index, prompt, partial_reasoning, has_boxed)
        for k in range(len(steps)):
            partial_reasoning = " ".join(steps[: k + 1])
            has_boxed = "\\boxed" in steps[k]
            prompt = f"{base_prompt}<think>\n{partial_reasoning}"
            prompt_data.append((k, prompt, partial_reasoning, has_boxed))

        # 4. Identify which steps need generation (skip if answer already visible)
        steps_needing_generation = [
            (k, prompt, partial, has_boxed)
            for k, prompt, partial, has_boxed in prompt_data
            if not has_boxed
        ]

        # 5. Batched generation for all steps that need it
        step_accuracies = [None] * len(steps)  # Pre-allocate
        generation_results = {}

        if steps_needing_generation:
            prompts_to_generate = [item[1] for item in steps_needing_generation]
            print(f"  Generating {len(prompts_to_generate)} step prompts in batch...")

            outputs = self.llm.generate(
                prompts_to_generate, self.params_handoff, use_tqdm=True
            )

            # Map results back to step indices
            for idx, (k, prompt, partial_reasoning, _) in enumerate(
                steps_needing_generation
            ):
                generation_results[k] = (outputs[idx], partial_reasoning)

        # 6. Process all results
        first_hsp_index = -1
        hsp_indices = []

        for k, prompt, partial_reasoning, has_boxed in prompt_data:
            if has_boxed:
                # Answer visible in prefix - auto success
                accuracy = 1.0
                print(
                    f"  Step {k}/{len(steps)-1}: Acc=1.00 (boxed in prefix) | Last: '{steps[k][-30:] if len(steps[k]) > 30 else steps[k]}'"
                )
            else:
                # Get generation results
                output, partial = generation_results[k]
                completions = [o.text for o in output.outputs]

                correct_count = sum(
                    1
                    for comp in completions
                    if MathVerifier.is_correct(partial + comp, ground_truth)
                )
                accuracy = correct_count / N_SAMPLES
                print(
                    f"  Step {k}/{len(steps)-1}: Acc={accuracy:.2f} | Last: '{steps[k][-30:] if len(steps[k]) > 30 else steps[k]}'"
                )

            step_accuracies[k] = accuracy

            # Check HSP
            if accuracy >= THRESHOLD_TAU:
                hsp_indices.append(k)
                if first_hsp_index == -1:
                    first_hsp_index = k
                    print(f"  [HSP FOUND] at k={k}")

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
            "hsp_indices": hsp_indices,
            "hsp_step_accuracies": step_accuracies,
        }

    def run_experiment(self):
        """Run the full experiment."""
        print("Loading ESSP Results...")
        dataset = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass

        print(f"Loaded {len(dataset)} problems.")

        # Open output file
        with open(OUTPUT_FILE, "w") as f_out:
            for i, data in enumerate(dataset):
                # Only process problems that the Senior actually solved (ESSP != -1)
                # If Senior failed, Handoff is undefined/moot.
                if data.get("first_essp_index") == -1:
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