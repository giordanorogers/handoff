"""scripts/attention_pruning_experiment.py

Prune sentences based on attention attribution scores (lowest first) and
evaluate handoff accuracy at each pruning step.

Compares attention-guided pruning against random baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.attention_pruning_experiment --problem_index 0
    CUDA_VISIBLE_DEVICES=1 python -m scripts.attention_pruning_experiment --problem_index 1
    CUDA_VISIBLE_DEVICES=2 python -m scripts.attention_pruning_experiment --problem_index 2
"""

import os
import sys
import json
import argparse
import logging
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vllm import LLM, SamplingParams

from src.utils import MathVerifier

SEED = 9001
MODEL_NAME = "Qwen/Qwen3-4B"
HSP_DATA_FILE = "data/hsp_step_accuracies.jsonl"
ATTENTION_DIR = "data/attention_attribution_results"
OUTPUT_DIR = "data/attention_pruning_results"
LOG_DIR = "logs/attention_pruning"

N_SAMPLES = 20
THRESHOLD_TAU = 0.5
MAX_TOKENS_COT = 31_000

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)


def setup_logging(problem_index):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"problem_{problem_index}_{timestamp}.log")
    
    logger = logging.getLogger(f"attention_pruning_{problem_index}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to: {log_file}")
    
    return logger, log_file


class AttentionPruningExperiment:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info(f"Loading Model: {MODEL_NAME}...")
        
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
        
        self.params_handoff = SamplingParams(
            n=N_SAMPLES,
            temperature=0.6,
            top_p=0.90,
            max_tokens=MAX_TOKENS_COT,
            seed=SEED,
        )

    def _build_prompt(self, problem, partial_reasoning):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        base = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return f"{base}<think>\n{partial_reasoning}"

    def _evaluate_handoff(self, problem, sentences, ground_truth):
        """Evaluate handoff accuracy for a given subset of sentences."""
        if len(sentences) == 0:
            partial_reasoning = ""
        else:
            partial_reasoning = " ".join(sentences)
        
        prompt = self._build_prompt(problem, partial_reasoning)
        
        outputs = self.llm.generate([prompt], self.params_handoff, use_tqdm=False)
        completions = [o.text for o in outputs[0].outputs]
        
        correct_count = 0
        for comp in completions:
            try:
                is_correct = MathVerifier.is_correct(partial_reasoning + comp, ground_truth)
                if is_correct:
                    correct_count += 1
            except Exception as e:
                self.logger.warning(f"MathVerifier error: {e}")
                continue
        
        return correct_count / N_SAMPLES

    def load_attention_scores(self, problem_index):
        """Load attention attribution results for a problem."""
        attention_file = os.path.join(ATTENTION_DIR, f"problem_{problem_index}_attention.json")
        
        if not os.path.exists(attention_file):
            self.logger.error(f"Attention file not found: {attention_file}")
            return None
        
        with open(attention_file, "r") as f:
            return json.load(f)

    def run_experiment(self, hsp_data, attention_data, problem_index):
        """
        Run attention-guided pruning experiment.
        
        Prunes sentences in order of lowest attention first, evaluating
        handoff accuracy at each step.
        """
        problem = hsp_data["problem"]
        ground_truth = hsp_data["solution"]
        steps = hsp_data["steps"]
        hsp_index = hsp_data["first_hsp_index"]
        problem_id = hsp_data.get("id", f"problem_{problem_index}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"ATTENTION PRUNING EXPERIMENT - Problem {problem_id}")
        self.logger.info("=" * 60)
        
        # Validate HSP
        if hsp_index is None or hsp_index == -1:
            self.logger.warning(f"No valid HSP found (hsp_index={hsp_index}), skipping.")
            return None
        
        # Get attention scores for sentences 0 to HSP
        # Attention was computed for 0 to ESSP, so we filter to 0 to HSP
        attention_scores = attention_data["normalized_scores"]
        essp_index = attention_data["essp_index"]
        
        self.logger.info(f"HSP index: {hsp_index}")
        self.logger.info(f"ESSP index (attention computed up to): {essp_index}")
        self.logger.info(f"Total steps in trace: {len(steps)}")
        
        if hsp_index > essp_index:
            self.logger.error(f"HSP ({hsp_index}) > ESSP ({essp_index}), cannot proceed.")
            return None
        
        # Filter to sentences 0 to HSP (inclusive)
        hsp_attention_scores = attention_scores[:hsp_index + 1]
        hsp_sentences = steps[:hsp_index + 1]
        
        self.logger.info(f"Sentences for pruning (0 to HSP): {len(hsp_sentences)}")
        
        # Create ranking: sort sentence indices by attention (ascending = lowest first)
        sentence_indices = list(range(len(hsp_sentences)))
        sorted_by_attention = sorted(
            sentence_indices, 
            key=lambda i: hsp_attention_scores[i]
        )
        
        self.logger.info("")
        self.logger.info("Attention ranking (lowest to highest):")
        for rank, idx in enumerate(sorted_by_attention[:10]):
            score = hsp_attention_scores[idx]
            preview = hsp_sentences[idx][:60].replace('\n', ' ')
            self.logger.info(f"  Rank {rank}: idx={idx}, attn={score:.4f}, '{preview}...'")
        if len(sorted_by_attention) > 10:
            self.logger.info(f"  ... ({len(sorted_by_attention) - 10} more)")
        
        # === ATTENTION-GUIDED PRUNING ===
        self.logger.info("")
        self.logger.info("=" * 40)
        self.logger.info("ATTENTION-GUIDED PRUNING (lowest first)")
        self.logger.info("=" * 40)
        
        attention_results = self._run_pruning_sequence(
            problem, ground_truth, hsp_sentences, 
            sorted_by_attention, hsp_attention_scores,
            "attention"
        )
        
        # === RANDOM BASELINE ===
        self.logger.info("")
        self.logger.info("=" * 40)
        self.logger.info("RANDOM BASELINE PRUNING")
        self.logger.info("=" * 40)
        
        random.seed(SEED)
        random_order = sentence_indices.copy()
        random.shuffle(random_order)
        
        random_results = self._run_pruning_sequence(
            problem, ground_truth, hsp_sentences,
            random_order, hsp_attention_scores,
            "random"
        )
        
        # === COMPILE RESULTS ===
        result = {
            "problem_id": problem_id,
            "problem_index": problem_index,
            "hsp_index": hsp_index,
            "essp_index": essp_index,
            "num_sentences": len(hsp_sentences),
            "attention_scores": hsp_attention_scores,
            "attention_pruning_order": sorted_by_attention,
            "random_pruning_order": random_order,
            "attention_results": attention_results,
            "random_results": random_results,
        }
        
        # Save results
        self._save_results(result, problem_index)
        
        # Generate visualizations
        self._generate_visualizations(result, problem_index)
        
        return result

    def _run_pruning_sequence(self, problem, ground_truth, sentences, 
                               pruning_order, attention_scores, method_name):
        """
        Run pruning sequence and evaluate accuracy at each step.
        
        Returns list of dicts with accuracy at each pruning step.
        """
        results = []
        current_indices = set(range(len(sentences)))
        
        # Baseline: all sentences
        self.logger.info(f"Step 0: Evaluating baseline ({len(current_indices)} sentences)...")
        current_sentences = [sentences[i] for i in sorted(current_indices)]
        baseline_acc = self._evaluate_handoff(problem, current_sentences, ground_truth)
        
        results.append({
            "step": 0,
            "removed_index": None,
            "removed_attention": None,
            "remaining_count": len(current_indices),
            "remaining_indices": sorted(current_indices),
            "accuracy": baseline_acc,
        })
        self.logger.info(f"  Baseline accuracy: {baseline_acc:.2f}")
        
        # Iteratively prune
        for step, idx_to_remove in enumerate(pruning_order, start=1):
            if idx_to_remove not in current_indices:
                continue
            
            current_indices.remove(idx_to_remove)
            
            if len(current_indices) == 0:
                # Edge case: all sentences removed
                acc = self._evaluate_handoff(problem, [], ground_truth)
            else:
                current_sentences = [sentences[i] for i in sorted(current_indices)]
                acc = self._evaluate_handoff(problem, current_sentences, ground_truth)
            
            removed_attn = attention_scores[idx_to_remove]
            preview = sentences[idx_to_remove][:50].replace('\n', ' ')
            
            results.append({
                "step": step,
                "removed_index": idx_to_remove,
                "removed_attention": removed_attn,
                "remaining_count": len(current_indices),
                "remaining_indices": sorted(current_indices),
                "accuracy": acc,
            })
            
            self.logger.info(
                f"  Step {step}: Removed idx={idx_to_remove} (attn={removed_attn:.4f}), "
                f"remaining={len(current_indices)}, acc={acc:.2f}"
            )
            
            # Early stopping if accuracy drops to 0 and stays there
            if acc == 0 and len(current_indices) < len(sentences) // 2:
                self.logger.info(f"  Early stopping: accuracy at 0 with {len(current_indices)} remaining")
                # Fill remaining steps with 0 accuracy
                for remaining_idx in pruning_order[step:]:
                    if remaining_idx in current_indices:
                        current_indices.remove(remaining_idx)
                        results.append({
                            "step": len(results),
                            "removed_index": remaining_idx,
                            "removed_attention": attention_scores[remaining_idx],
                            "remaining_count": len(current_indices),
                            "remaining_indices": sorted(current_indices),
                            "accuracy": 0.0,
                        })
                break
        
        return results

    def _save_results(self, result, problem_index):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"problem_{problem_index}_pruning_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        self.logger.info(f"Results saved: {output_file}")

    def _generate_visualizations(self, result, problem_index):
        """Generate comparison plots."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        attention_results = result["attention_results"]
        random_results = result["random_results"]
        num_sentences = result["num_sentences"]
        
        # Extract data for plotting
        attn_remaining = [r["remaining_count"] for r in attention_results]
        attn_acc = [r["accuracy"] for r in attention_results]
        
        rand_remaining = [r["remaining_count"] for r in random_results]
        rand_acc = [r["accuracy"] for r in random_results]
        
        # Plot 1: Accuracy vs Sentences Remaining
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(attn_remaining, attn_acc, 'b-o', label='Attention (lowest first)', markersize=4)
        ax1.plot(rand_remaining, rand_acc, 'r--s', label='Random', markersize=4, alpha=0.7)
        ax1.axhline(y=THRESHOLD_TAU, color='gray', linestyle=':', label=f'Threshold (τ={THRESHOLD_TAU})')
        ax1.set_xlabel('Sentences Remaining')
        ax1.set_ylabel('Handoff Accuracy')
        ax1.set_title(f'Problem {problem_index}: Accuracy vs Sentences Remaining')
        ax1.legend()
        ax1.set_xlim(num_sentences + 1, -1)  # Reverse x-axis
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Fraction Removed
        ax2 = axes[1]
        attn_frac_removed = [1 - r["remaining_count"]/num_sentences for r in attention_results]
        rand_frac_removed = [1 - r["remaining_count"]/num_sentences for r in random_results]
        
        ax2.plot(attn_frac_removed, attn_acc, 'b-o', label='Attention (lowest first)', markersize=4)
        ax2.plot(rand_frac_removed, rand_acc, 'r--s', label='Random', markersize=4, alpha=0.7)
        ax2.axhline(y=THRESHOLD_TAU, color='gray', linestyle=':', label=f'Threshold (τ={THRESHOLD_TAU})')
        ax2.set_xlabel('Fraction of Sentences Removed')
        ax2.set_ylabel('Handoff Accuracy')
        ax2.set_title(f'Problem {problem_index}: Accuracy vs Fraction Removed')
        ax2.legend()
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(OUTPUT_DIR, f"problem_{problem_index}_pruning_curves.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot saved: {plot_file}")
        
        # Plot 3: Attention score distribution with pruning threshold
        fig, ax = plt.subplots(figsize=(10, 4))
        
        attention_scores = result["attention_scores"]
        sorted_scores = sorted(enumerate(attention_scores), key=lambda x: x[1])
        
        indices = [x[0] for x in sorted_scores]
        scores = [x[1] for x in sorted_scores]
        
        # Color by whether sentence was "essential" (accuracy dropped significantly when removed)
        # For now, just show the distribution
        ax.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
        ax.set_xlabel('Sentence (sorted by attention)')
        ax.set_ylabel('Normalized Attention Score')
        ax.set_title(f'Problem {problem_index}: Attention Score Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sentence indices as x-tick labels (sparse)
        tick_positions = list(range(0, len(scores), max(1, len(scores)//10)))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([indices[i] for i in tick_positions], fontsize=8)
        
        plt.tight_layout()
        
        dist_file = os.path.join(OUTPUT_DIR, f"problem_{problem_index}_attention_distribution.png")
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Distribution plot saved: {dist_file}")
        
        # Summary statistics
        self.logger.info("")
        self.logger.info("=" * 40)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 40)
        
        # Find where accuracy drops below threshold
        attn_threshold_idx = next(
            (i for i, r in enumerate(attention_results) if r["accuracy"] < THRESHOLD_TAU),
            len(attention_results)
        )
        rand_threshold_idx = next(
            (i for i, r in enumerate(random_results) if r["accuracy"] < THRESHOLD_TAU),
            len(random_results)
        )
        
        if attn_threshold_idx > 0:
            attn_removable = attention_results[attn_threshold_idx - 1]["remaining_count"]
            attn_removed = num_sentences - attn_removable
        else:
            attn_removable = num_sentences
            attn_removed = 0
            
        if rand_threshold_idx > 0:
            rand_removable = random_results[rand_threshold_idx - 1]["remaining_count"]
            rand_removed = num_sentences - rand_removable
        else:
            rand_removable = num_sentences
            rand_removed = 0
        
        self.logger.info(f"Total sentences: {num_sentences}")
        self.logger.info(f"Attention pruning: Can remove {attn_removed} sentences before dropping below τ={THRESHOLD_TAU}")
        self.logger.info(f"  -> Minimum needed: {attn_removable} sentences ({100*attn_removable/num_sentences:.1f}%)")
        self.logger.info(f"Random pruning: Can remove {rand_removed} sentences before dropping below τ={THRESHOLD_TAU}")
        self.logger.info(f"  -> Minimum needed: {rand_removable} sentences ({100*rand_removable/num_sentences:.1f}%)")
        
        # Area under curve comparison
        attn_auc = np.trapz(attn_acc, attn_remaining) / num_sentences if len(attn_remaining) > 1 else 0
        rand_auc = np.trapz(rand_acc, rand_remaining) / num_sentences if len(rand_remaining) > 1 else 0
        
        self.logger.info(f"Normalized AUC (attention): {attn_auc:.3f}")
        self.logger.info(f"Normalized AUC (random): {rand_auc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Attention-guided pruning experiment")
    parser.add_argument("--problem_index", type=int, required=True,
                        help="Index of problem to process (0, 1, or 2)")
    args = parser.parse_args()
    
    logger, log_file = setup_logging(args.problem_index)
    
    logger.info("=" * 60)
    logger.info("ATTENTION-GUIDED PRUNING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Problem index: {args.problem_index}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"N_SAMPLES: {N_SAMPLES}")
    logger.info(f"THRESHOLD_TAU: {THRESHOLD_TAU}")
    logger.info(f"SEED: {SEED}")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load HSP data
    logger.info(f"Loading HSP data from {HSP_DATA_FILE}...")
    if not os.path.exists(HSP_DATA_FILE):
        logger.error(f"HSP data file not found: {HSP_DATA_FILE}")
        sys.exit(1)
    
    hsp_dataset = []
    with open(HSP_DATA_FILE, "r") as f:
        for line in f:
            if line.strip():
                hsp_dataset.append(json.loads(line))
    
    if args.problem_index >= len(hsp_dataset):
        logger.error(f"Problem index {args.problem_index} out of range (have {len(hsp_dataset)} problems)")
        sys.exit(1)
    
    hsp_data = hsp_dataset[args.problem_index]
    
    # Load attention data
    experiment = AttentionPruningExperiment(logger)
    attention_data = experiment.load_attention_scores(args.problem_index)
    
    if attention_data is None:
        logger.error("Failed to load attention data")
        sys.exit(1)
    
    # Run experiment
    result = experiment.run_experiment(hsp_data, attention_data, args.problem_index)
    
    if result:
        logger.info("")
        logger.info("=" * 60)
        logger.info("COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    else:
        logger.error("Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()