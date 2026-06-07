"""scripts/greedy_backward_elimination.py

Greedy backward elimination to find minimal sufficient subset for HSP.
Run on a single problem (specify via PROBLEM_INDEX), parallelizable across GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/greedy_backward_elimination.py --problem_index 0
    CUDA_VISIBLE_DEVICES=2 python -m scripts.greedy_backward_elimination --problem_index 1
    CUDA_VISIBLE_DEVICES=3 python -m scripts.greedy_backward_elimination --problem_index 2
"""

import os
import sys
import json
import argparse
import logging
import torch
import random
import numpy as np
from datetime import datetime
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

SEED = 9001
MODEL_NAME = "Qwen/Qwen3-4B"
INPUT_FILE = "data/hsp_step_accuracies.jsonl"
OUTPUT_DIR = "data/greedy_elimination_results"
LOG_DIR = "logs/greedy_elimination"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


def setup_logging(problem_index):
    """Set up logging to both file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"problem_{problem_index}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger(f"greedy_elimination_{problem_index}")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler - captures everything
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger, log_file


class GreedyBackwardElimination:
    """
    Greedy backward elimination to find minimal sufficient subset for handoff.
    
    Algorithm:
    1. Start with full prefix (sentences 0 to HSP inclusive)
    2. For each remaining sentence, test accuracy if removed
    3. Remove the sentence whose removal causes least accuracy drop
    4. Repeat until accuracy < threshold or no sentences can be safely removed
    """

    def __init__(self, logger):
        self.logger = logger
        self.logger.info(f"Loading Model: {MODEL_NAME}...")
        
        try:
            self.llm = LLM(
                model=MODEL_NAME,
                tensor_parallel_size=1,
                enable_prefix_caching=True,
                seed=SEED,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

        self.params_handoff = SamplingParams(
            n=N_SAMPLES,
            temperature=0.6,
            top_p=0.90,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED,
        )

    def _build_prompt(self, problem, partial_reasoning):
        """Build prompt with partial reasoning prefix."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        base = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return f"{base}<think>\n{partial_reasoning}"

    def _evaluate_handoff(self, problem, sentences, ground_truth):
        """
        Evaluate handoff accuracy for a given subset of sentences.
        Returns accuracy over N_SAMPLES rollouts.
        """
        if len(sentences) == 0:
            self.logger.warning("Empty sentence list passed to _evaluate_handoff")
            return 0.0
            
        partial_reasoning = " ".join(sentences)
        prompt = self._build_prompt(problem, partial_reasoning)
        
        try:
            outputs = self.llm.generate([prompt], self.params_handoff, use_tqdm=False)
            completions = [o.text for o in outputs[0].outputs]
        except Exception as e:
            self.logger.error(f"Generation failed in _evaluate_handoff: {e}")
            raise
        
        correct_count = 0
        for comp in completions:
            try:
                is_correct = MathVerifier.is_correct(partial_reasoning + comp, ground_truth)
                if is_correct:
                    correct_count += 1
            except Exception as e:
                self.logger.warning(f"MathVerifier.is_correct raised exception: {e}")
                # Treat as incorrect
                continue
                
        return correct_count / N_SAMPLES

    def _evaluate_batch(self, problem, sentence_subsets, ground_truth):
        """
        Evaluate handoff accuracy for multiple sentence subsets in batch.
        Returns list of accuracies corresponding to each subset.
        """
        if len(sentence_subsets) == 0:
            self.logger.warning("Empty sentence_subsets passed to _evaluate_batch")
            return []
        
        prompts = []
        partial_reasonings = []
        
        for sentences in sentence_subsets:
            if len(sentences) == 0:
                # Edge case: empty subset
                partial = ""
            else:
                partial = " ".join(sentences)
            partial_reasonings.append(partial)
            prompts.append(self._build_prompt(problem, partial))
        
        self.logger.debug(f"Batch evaluating {len(prompts)} prompts...")
        
        try:
            outputs = self.llm.generate(prompts, self.params_handoff, use_tqdm=True)
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise
        
        accuracies = []
        for idx, output in enumerate(outputs):
            completions = [o.text for o in output.outputs]
            partial = partial_reasonings[idx]
            
            correct_count = 0
            for comp in completions:
                try:
                    is_correct = MathVerifier.is_correct(partial + comp, ground_truth)
                    if is_correct:
                        correct_count += 1
                except Exception as e:
                    self.logger.warning(f"MathVerifier error at idx {idx}: {e}")
                    continue
                    
            accuracies.append(correct_count / N_SAMPLES)
        
        return accuracies

    def _save_checkpoint(self, result, problem_index, checkpoint_name):
        """Save intermediate checkpoint."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        checkpoint_file = os.path.join(
            OUTPUT_DIR,
            f"problem_{problem_index}_{checkpoint_name}.json"
        )
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(result, f, indent=2)
            self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def run_elimination(self, data, problem_index):
        """
        Run greedy backward elimination for a single problem.
        """
        problem = data["problem"]
        ground_truth = data["solution"]
        steps = data["steps"]
        hsp_index = data["first_hsp_index"]
        problem_id = data.get("id", "unknown")
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting elimination for Problem {problem_id}")
        self.logger.info(f"=" * 60)
        
        # Validate inputs
        if hsp_index is None or hsp_index == -1:
            self.logger.warning(f"Problem {problem_id}: No valid HSP found (hsp_index={hsp_index}), skipping.")
            return None
        
        if steps is None or len(steps) == 0:
            self.logger.error(f"Problem {problem_id}: No steps found.")
            return None
            
        if hsp_index >= len(steps):
            self.logger.error(f"Problem {problem_id}: HSP index {hsp_index} >= len(steps) {len(steps)}")
            return None
        
        # Start with all sentences up to and including HSP
        initial_indices = list(range(hsp_index + 1))
        current_indices = initial_indices.copy()
        
        self.logger.info(f"Problem ID: {problem_id}")
        self.logger.info(f"HSP Index: {hsp_index}")
        self.logger.info(f"Total steps in trace: {len(steps)}")
        self.logger.info(f"Initial sentences (0 to HSP): {len(current_indices)}")
        self.logger.info(f"Ground truth solution: {ground_truth[:100]}...")
        
        # Log first few sentences for sanity check
        self.logger.debug("First 3 sentences:")
        for i in range(min(3, len(current_indices))):
            self.logger.debug(f"  [{i}]: {steps[i][:80]}...")
        
        # Verify baseline accuracy
        self.logger.info("Evaluating baseline accuracy...")
        baseline_sentences = [steps[i] for i in current_indices]
        baseline_acc = self._evaluate_handoff(problem, baseline_sentences, ground_truth)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.2f}")
        
        if baseline_acc < THRESHOLD_TAU:
            self.logger.warning(f"Baseline accuracy {baseline_acc:.2f} < threshold {THRESHOLD_TAU}, cannot proceed.")
            result = {
                "problem_id": problem_id,
                "hsp_index": hsp_index,
                "initial_indices": initial_indices,
                "final_indices": current_indices,
                "baseline_accuracy": baseline_acc,
                "elimination_history": [],
                "status": "baseline_below_threshold",
            }
            self._save_checkpoint(result, problem_index, "final")
            return result
        
        elimination_history = []
        iteration = 0
        
        while len(current_indices) > 1:
            iteration += 1
            self.logger.info(f"")
            self.logger.info(f"--- Iteration {iteration} | Sentences remaining: {len(current_indices)} ---")
            
            # Test removing each sentence
            candidate_subsets = []
            candidate_removed_indices = []
            
            for idx_to_remove in current_indices:
                subset_indices = [idx for idx in current_indices if idx != idx_to_remove]
                subset_sentences = [steps[idx] for idx in subset_indices]
                candidate_subsets.append(subset_sentences)
                candidate_removed_indices.append(idx_to_remove)
            
            self.logger.info(f"Testing {len(candidate_subsets)} candidate removals...")
            
            # Batch evaluate all candidates
            accuracies = self._evaluate_batch(problem, candidate_subsets, ground_truth)
            
            if len(accuracies) != len(candidate_removed_indices):
                self.logger.error(f"Mismatch: {len(accuracies)} accuracies vs {len(candidate_removed_indices)} candidates")
                break
            
            # Log results for this iteration
            iteration_results = []
            for removed_idx, acc in zip(candidate_removed_indices, accuracies):
                truncated_sentence = steps[removed_idx][:100].replace('\n', ' ')
                iteration_results.append({
                    "removed_index": removed_idx,
                    "removed_sentence_preview": truncated_sentence,
                    "accuracy": acc,
                })
                self.logger.info(f"  Remove idx {removed_idx:3d}: acc={acc:.2f} | '{truncated_sentence[:50]}...'")
            
            # Find best candidate (highest accuracy after removal)
            best_idx_in_list = int(np.argmax(accuracies))
            best_acc = accuracies[best_idx_in_list]
            best_removed = candidate_removed_indices[best_idx_in_list]
            
            self.logger.info(f"Best removal candidate: idx {best_removed} with acc={best_acc:.2f}")
            
            # Check if we can safely remove this sentence
            if best_acc < THRESHOLD_TAU:
                self.logger.info(f"Cannot remove any sentence without dropping below threshold {THRESHOLD_TAU}.")
                elimination_history.append({
                    "iteration": iteration,
                    "results": iteration_results,
                    "removed": None,
                    "reason": "all_removals_below_threshold",
                })
                break
            
            # Remove the best candidate
            current_indices = [idx for idx in current_indices if idx != best_removed]
            elimination_history.append({
                "iteration": iteration,
                "results": iteration_results,
                "removed": best_removed,
                "removed_sentence_preview": steps[best_removed][:100].replace('\n', ' '),
                "accuracy_after": best_acc,
                "remaining_count": len(current_indices),
            })
            
            self.logger.info(f"REMOVED sentence {best_removed}. Remaining: {len(current_indices)} sentences")
            
            # Save checkpoint after each iteration
            checkpoint_result = {
                "problem_id": problem_id,
                "hsp_index": hsp_index,
                "initial_indices": initial_indices,
                "current_indices": current_indices,
                "baseline_accuracy": baseline_acc,
                "iteration": iteration,
                "elimination_history": elimination_history,
                "status": "in_progress",
            }
            self._save_checkpoint(checkpoint_result, problem_index, f"iter_{iteration:03d}")
        
        # Final evaluation
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 60)
        
        final_sentences = [steps[i] for i in current_indices]
        final_acc = self._evaluate_handoff(problem, final_sentences, ground_truth)
        
        compression_ratio = len(current_indices) / len(initial_indices) if len(initial_indices) > 0 else 0
        
        self.logger.info(f"Initial sentences: {len(initial_indices)}")
        self.logger.info(f"Final sentences: {len(current_indices)} ({100*compression_ratio:.1f}%)")
        self.logger.info(f"Baseline accuracy: {baseline_acc:.2f}")
        self.logger.info(f"Final accuracy: {final_acc:.2f}")
        self.logger.info(f"Final indices: {current_indices}")
        
        self.logger.info("")
        self.logger.info("Final sentences content:")
        for i, idx in enumerate(current_indices):
            self.logger.info(f"  [{idx}]: {steps[idx][:100].replace(chr(10), ' ')}...")
        
        result = {
            "problem_id": problem_id,
            "hsp_index": hsp_index,
            "total_steps": len(steps),
            "initial_indices": initial_indices,
            "final_indices": current_indices,
            "final_sentences": final_sentences,
            "baseline_accuracy": baseline_acc,
            "final_accuracy": final_acc,
            "compression_ratio": compression_ratio,
            "total_iterations": iteration,
            "elimination_history": elimination_history,
            "status": "completed",
        }
        
        self._save_checkpoint(result, problem_index, "final")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Greedy backward elimination for HSP")
    parser.add_argument("--problem_index", type=int, required=True,
                        help="Index of problem to process (0, 1, or 2)")
    args = parser.parse_args()
    
    # Set up logging first
    logger, log_file = setup_logging(args.problem_index)
    
    logger.info("=" * 60)
    logger.info("GREEDY BACKWARD ELIMINATION")
    logger.info("=" * 60)
    logger.info(f"Problem index: {args.problem_index}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"N_SAMPLES: {N_SAMPLES}")
    logger.info(f"THRESHOLD_TAU: {THRESHOLD_TAU}")
    logger.info(f"MAX_TOKENS_COT: {MAX_TOKENS_COT}")
    logger.info(f"SEED: {SEED}")
    
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logger.info("Random seeds set.")
    
    # Load dataset
    logger.info(f"Loading data from {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    dataset = []
    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
    
    logger.info(f"Loaded {len(dataset)} problems.")
    
    if args.problem_index >= len(dataset):
        logger.error(f"Problem index {args.problem_index} out of range (dataset has {len(dataset)} problems).")
        sys.exit(1)
    
    if args.problem_index < 0:
        logger.error(f"Problem index must be non-negative.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get the problem data
    data = dataset[args.problem_index]
    
    # Log problem info
    logger.info(f"Problem ID: {data.get('id', 'unknown')}")
    logger.info(f"HSP Index: {data.get('first_hsp_index', 'N/A')}")
    logger.info(f"ESSP Index: {data.get('first_essp_index', 'N/A')}")
    logger.info(f"Total steps: {len(data.get('steps', []))}")
    
    # Validate required fields
    required_fields = ["problem", "solution", "steps", "first_hsp_index"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            sys.exit(1)
    
    # Run elimination
    try:
        experiment = GreedyBackwardElimination(logger)
        result = experiment.run_elimination(data, args.problem_index)
    except Exception as e:
        logger.exception(f"Fatal error during elimination: {e}")
        sys.exit(1)
    
    if result:
        # Save final result with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            OUTPUT_DIR,
            f"problem_{args.problem_index}_result_{timestamp}.json"
        )
        
        try:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Final result saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save final result: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()