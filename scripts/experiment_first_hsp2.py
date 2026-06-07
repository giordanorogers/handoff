import os
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

INPUT_FILE = "data/MATH_92/MATH_92_indexed.jsonl"
CHECKPOINT_FILE = "data/MATH_92/MATH_92_hsp.jsonl"
OUTPUT_FILE = "data/experiment_2_hsp_results_focused.jsonl"

N_SAMPLES_QUICK = 5
N_SAMPLES_FULL = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

COARSE_STRIDE = 10

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


class HSPExperimentOptimized:
    """Optimized HSP experiment using coarse-to-fine search with quick validation."""

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

        self.params_quick = SamplingParams(
            n=N_SAMPLES_QUICK,
            temperature=0.7,
            top_p=0.95,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED,
        )
        
        self.params_full = SamplingParams(
            n=N_SAMPLES_FULL,
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
        outputs = self.llm.generate([base_prompt], self.params_full, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]

        correct_count = sum(
            1 for comp in completions if MathVerifier.is_correct(comp, ground_truth)
        )
        return correct_count / N_SAMPLES_FULL

    def _check_step(self, k, steps, base_prompt, ground_truth, quick=True):
        """Check accuracy at step k.
        
        Args:
            k: Step index
            steps: List of reasoning steps
            base_prompt: Base prompt string
            ground_truth: Ground truth answer
            quick: If True, use fewer samples for speed
            
        Returns:
            Accuracy at this step
        """
        if "\\boxed" in steps[k]:
            return 1.0
            
        partial = " ".join(steps[:k+1])
        prompt = f"{base_prompt}<think>\n{partial}"
        
        params = self.params_quick if quick else self.params_full
        n_samples = N_SAMPLES_QUICK if quick else N_SAMPLES_FULL
        
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]
        
        correct_count = sum(
            1 for comp in completions 
            if MathVerifier.is_correct(partial + comp, ground_truth)
        )
        return correct_count / n_samples

    def _batch_check_steps(self, indices, steps, base_prompt, ground_truth, quick=True):
        """Batch check multiple steps at once.
        
        Args:
            indices: List of step indices to check
            steps: List of reasoning steps
            base_prompt: Base prompt string
            ground_truth: Ground truth answer
            quick: If True, use fewer samples
            
        Returns:
            Dict mapping index -> accuracy
        """
        params = self.params_quick if quick else self.params_full
        n_samples = N_SAMPLES_QUICK if quick else N_SAMPLES_FULL
        
        results = {}
        prompts_to_generate = []
        indices_to_generate = []
        partials = {}
        
        for k in indices:
            if "\\boxed" in steps[k]:
                results[k] = 1.0
            else:
                partial = " ".join(steps[:k+1])
                prompt = f"{base_prompt}<think>\n{partial}"
                prompts_to_generate.append(prompt)
                indices_to_generate.append(k)
                partials[k] = partial
        
        if prompts_to_generate:
            outputs = self.llm.generate(prompts_to_generate, params, use_tqdm=False)
            
            for i, k in enumerate(indices_to_generate):
                completions = [o.text for o in outputs[i].outputs]
                correct_count = sum(
                    1 for comp in completions
                    if MathVerifier.is_correct(partials[k] + comp, ground_truth)
                )
                results[k] = correct_count / n_samples
        
        return results

    def find_hsp_coarse_fine(self, data, coarse_stride=COARSE_STRIDE):
        """Find first HSP using coarse-to-fine search.
        
        Strategy:
        1. Coarse scan: Check every `coarse_stride` steps with quick samples
        2. Fine search: Once we find first coarse success, search the window before it
        3. Validate: Confirm the found HSP with full samples
        
        This reduces O(n) to roughly O(n/stride + stride) checks.
        """
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]
        n_steps = len(steps)

        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {n_steps}")

        base_prompt = self._build_base_prompt(problem)

        # Verify Junior cannot solve on its own
        base_accuracy = self._check_base_accuracy(base_prompt, ground_truth)
        print(f"  Base Acc={base_accuracy:.2f}")

        if base_accuracy > 0.0:
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": -1,
                "skipped_reason": "junior_baseline_nonzero",
            }

        # Build coarse indices
        coarse_indices = list(range(0, n_steps, coarse_stride))
        if coarse_indices[-1] != n_steps - 1:
            coarse_indices.append(n_steps - 1)

        print(f"  Coarse scan: {len(coarse_indices)} checkpoints (stride={coarse_stride})")

        # Batch coarse scan with quick samples
        coarse_results = self._batch_check_steps(
            coarse_indices, steps, base_prompt, ground_truth, quick=True
        )
        
        # Find first coarse success
        first_coarse_idx = None
        for i, k in enumerate(coarse_indices):
            acc = coarse_results[k]
            status = "✓" if acc >= THRESHOLD_TAU else "✗"
            print(f"    Coarse [{i}] step {k}: acc={acc:.2f} {status}")
            if acc >= THRESHOLD_TAU and first_coarse_idx is None:
                first_coarse_idx = i
                break  # Stop printing after first success

        if first_coarse_idx is None:
            print("  [FAILURE] No coarse success found.")
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": -1,
            }

        first_coarse_k = coarse_indices[first_coarse_idx]
        print(f"  First coarse success at step {first_coarse_k}")

        # Fine search: check steps between previous coarse point and first success
        if first_coarse_idx == 0:
            fine_start = 0
        else:
            fine_start = coarse_indices[first_coarse_idx - 1] + 1
        fine_end = first_coarse_k

        fine_indices = list(range(fine_start, fine_end))
        
        if fine_indices:
            print(f"  Fine search: steps {fine_start} to {fine_end - 1}")
            
            # Batch fine search with quick samples
            fine_results = self._batch_check_steps(
                fine_indices, steps, base_prompt, ground_truth, quick=True
            )
            
            # Find first fine success
            for k in fine_indices:
                acc = fine_results[k]
                status = "✓" if acc >= THRESHOLD_TAU else "✗"
                print(f"    Fine step {k}: acc={acc:.2f} {status}")
                
                if acc >= THRESHOLD_TAU:
                    # Validate with full samples
                    full_acc = self._check_step(
                        k, steps, base_prompt, ground_truth, quick=False
                    )
                    print(f"    Validate step {k}: full_acc={full_acc:.2f}")
                    
                    if full_acc >= THRESHOLD_TAU:
                        print(f"  [FIRST HSP FOUND] at k={k}")
                        return {
                            "id": data.get("id"),
                            "problem": problem,
                            "solution": ground_truth,
                            "total_steps": n_steps,
                            "first_essp_index": data.get("first_essp_index"),
                            "essp_indices": data.get("essp_indices"),
                            "first_hsp_index": k,
                            "quick_acc": acc,
                            "validated_acc": full_acc,
                        }

        # Validate the coarse success point
        full_acc = self._check_step(
            first_coarse_k, steps, base_prompt, ground_truth, quick=False
        )
        print(f"  Validate coarse step {first_coarse_k}: full_acc={full_acc:.2f}")

        if full_acc >= THRESHOLD_TAU:
            print(f"  [FIRST HSP FOUND] at k={first_coarse_k}")
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": first_coarse_k,
                "quick_acc": coarse_results[first_coarse_k],
                "validated_acc": full_acc,
            }

        # Edge case: quick samples gave false positive, search forward
        print(f"  Coarse validation failed, searching forward...")
        
        forward_start = first_coarse_k + 1
        forward_end = coarse_indices[first_coarse_idx + 1] if first_coarse_idx + 1 < len(coarse_indices) else n_steps
        
        for k in range(forward_start, forward_end):
            acc = self._check_step(k, steps, base_prompt, ground_truth, quick=False)
            print(f"    Forward step {k}: acc={acc:.2f}")
            if acc >= THRESHOLD_TAU:
                print(f"  [FIRST HSP FOUND] at k={k}")
                return {
                    "id": data.get("id"),
                    "problem": problem,
                    "solution": ground_truth,
                    "total_steps": n_steps,
                    "first_essp_index": data.get("first_essp_index"),
                    "essp_indices": data.get("essp_indices"),
                    "first_hsp_index": k,
                    "validated_acc": acc,
                }

        print("  [FAILURE] Validation failed, no HSP found.")
        return {
            "id": data.get("id"),
            "problem": problem,
            "solution": ground_truth,
            "total_steps": n_steps,
            "first_essp_index": data.get("first_essp_index"),
            "essp_indices": data.get("essp_indices"),
            "first_hsp_index": -1,
        }

    def find_hsp_binary(self, data):
        """Find first HSP using binary search.
        
        Assumes accuracy is roughly monotonic with step index.
        Faster than coarse-fine for very long traces, but may miss
        the true first HSP if there's non-monotonicity.
        """
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]
        n_steps = len(steps)

        print(f"\nProblem ID: {data.get('id', 'N/A')} | Steps: {n_steps}")

        base_prompt = self._build_base_prompt(problem)

        # Verify Junior cannot solve on its own
        base_accuracy = self._check_base_accuracy(base_prompt, ground_truth)
        print(f"  Base Acc={base_accuracy:.2f}")

        if base_accuracy > 0.0:
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": -1,
                "skipped_reason": "junior_baseline_nonzero",
            }

        # Binary search
        lo, hi = 0, n_steps - 1
        first_hsp = -1
        checks = 0

        print(f"  Binary search over {n_steps} steps...")

        while lo <= hi:
            mid = (lo + hi) // 2
            acc = self._check_step(mid, steps, base_prompt, ground_truth, quick=True)
            checks += 1
            
            status = "✓" if acc >= THRESHOLD_TAU else "✗"
            print(f"    Check {checks}: step {mid}, acc={acc:.2f} {status}")

            if acc >= THRESHOLD_TAU:
                first_hsp = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if first_hsp == -1:
            print("  [FAILURE] Binary search found no HSP.")
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": -1,
            }

        # Validate with full samples
        full_acc = self._check_step(first_hsp, steps, base_prompt, ground_truth, quick=False)
        print(f"  Validate step {first_hsp}: full_acc={full_acc:.2f}")

        if full_acc >= THRESHOLD_TAU:
            print(f"  [FIRST HSP FOUND] at k={first_hsp} ({checks} checks)")
            return {
                "id": data.get("id"),
                "problem": problem,
                "solution": ground_truth,
                "total_steps": n_steps,
                "first_essp_index": data.get("first_essp_index"),
                "essp_indices": data.get("essp_indices"),
                "first_hsp_index": first_hsp,
                "validated_acc": full_acc,
                "binary_checks": checks,
            }

        # If validation fails, fall back to linear search around the candidate
        print(f"  Validation failed, linear search around step {first_hsp}...")
        
        search_radius = 5
        for k in range(max(0, first_hsp - search_radius), min(n_steps, first_hsp + search_radius + 1)):
            if k == first_hsp:
                continue
            acc = self._check_step(k, steps, base_prompt, ground_truth, quick=False)
            if acc >= THRESHOLD_TAU:
                print(f"  [FIRST HSP FOUND] at k={k}")
                return {
                    "id": data.get("id"),
                    "problem": problem,
                    "solution": ground_truth,
                    "total_steps": n_steps,
                    "first_essp_index": data.get("first_essp_index"),
                    "essp_indices": data.get("essp_indices"),
                    "first_hsp_index": k,
                    "validated_acc": acc,
                }

        print("  [FAILURE] No valid HSP found after fallback.")
        return {
            "id": data.get("id"),
            "problem": problem,
            "solution": ground_truth,
            "total_steps": n_steps,
            "first_essp_index": data.get("first_essp_index"),
            "essp_indices": data.get("essp_indices"),
            "first_hsp_index": -1,
        }

    def run_experiment(self, shard=0, num_shards=1, method="coarse_fine"):
        """Run the experiment on a shard of the data.
        
        Args:
            shard: Shard index for this worker
            num_shards: Total number of shards
            method: Search method - "coarse_fine" or "binary"
        """
        print(f"Loading data (method={method})...")
        dataset = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass

        checkpoint_ids = set()
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        checkpoint_ids.add(data["id"])
                    except:
                        pass

        # Filter to problems not in checkpoint
        dataset = [p for p in dataset if p["id"] not in checkpoint_ids]

        # Filter to this shard
        dataset = [p for i, p in enumerate(dataset) if i % num_shards == shard]
        print(f"Loaded {len(dataset)} problems for shard {shard}/{num_shards}.")

        # Output file per shard
        if num_shards > 1:
            output_file = OUTPUT_FILE.replace(".jsonl", f"_shard{shard}.jsonl")
        else:
            output_file = OUTPUT_FILE

        # Select search method
        if method == "binary":
            find_hsp = self.find_hsp_binary
        else:
            find_hsp = self.find_hsp_coarse_fine

        with open(output_file, "a") as f_out:  # Append mode for resumability
            for i, data in enumerate(dataset):
                if data.get("first_essp_index") == -1:
                    print(f"Skipping ID {data.get('id')}: Senior failed (ESSP=-1).")
                    continue

                result = find_hsp(data)

                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                
                print(f"  Progress: {i+1}/{len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--method", 
        type=str, 
        default="coarse_fine",
        choices=["coarse_fine", "binary"],
        help="Search method: coarse_fine (default) or binary"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Coarse stride for coarse_fine method"
    )
    args = parser.parse_args()

    # Update global stride if specified
    COARSE_STRIDE = args.stride

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    experiment = HSPExperimentOptimized(gpu_id=args.gpu)
    experiment.run_experiment(
        shard=args.shard, 
        num_shards=args.num_shards,
        method=args.method
    )