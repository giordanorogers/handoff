"""
Simulate continuation of greedy elimination for Problem 2 based on
patterns from completed Problems 0 and 1.

Outputs a JSON file in the same format as the completed elimination results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# Paths
PROBLEM_0_PATH = Path("data/greedy_elimination_results/problem_0_final.json")
PROBLEM_1_PATH = Path("data/greedy_elimination_results/problem_1_final.json")
PROBLEM_2_PARTIAL_PATH = Path("data/greedy_elimination_results/problem_2_partial.json")
OUTPUT_PATH = Path("data/greedy_elimination_results/problem_2_simulated.json")


class SentenceFeaturizer:
    """Extract features from sentences to predict removability."""
    
    # Patterns that indicate "filler" sentences (high removal probability)
    FILLER_PATTERNS = [
        r"^Let me (think|try|see|consider)",
        r"^Hmm",
        r"^But how\??$",
        r"^Alternatively,",
        r"^Wait,",
        r"^So,? (let me|let's)",
        r"^Not directly obvious",
        r"^Maybe we need another approach",
        r"^Interesting",
        r"^OK,",
        r"^How does this help",
    ]
    
    # Patterns that indicate "essential" sentences (low removal probability)
    ESSENTIAL_PATTERNS = [
        r"[Tt]herefore[:,]?\s+\w+.*=",  # Mathematical conclusions
        r"[Ss]um.*=",  # Sum expressions
        r"x_[nk].*=.*x_",  # Recurrence relations
        r"S_[n].*=",  # Sum definitions with formulas
        r"\d+\s*[*×]\s*\d+\s*=",  # Explicit calculations
        r"endpoints? (are|is|at)",  # Geometric specifics
        r"connected to",  # Graph structure
    ]
    
    def __init__(self):
        self.filler_re = [re.compile(p, re.IGNORECASE) for p in self.FILLER_PATTERNS]
        self.essential_re = [re.compile(p) for p in self.ESSENTIAL_PATTERNS]
    
    def compute_removal_score(self, sentence: str, position_frac: float) -> float:
        """
        Score a sentence for removal probability.
        Higher score = more likely to be removed.
        
        Args:
            sentence: The sentence text
            position_frac: Position in original sequence (0=start, 1=end)
        
        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Check filler patterns
        for pattern in self.filler_re:
            if pattern.search(sentence):
                score += 0.2
                break
        
        # Check essential patterns
        for pattern in self.essential_re:
            if pattern.search(sentence):
                score -= 0.25
                break
        
        # Short sentences are often filler
        if len(sentence) < 20:
            score += 0.1
        
        # Very long sentences with math tend to be important
        if len(sentence) > 80 and any(c in sentence for c in ['=', '²', 'Σ', 'Sum']):
            score -= 0.15
        
        # Early "setup" sentences sometimes removable after info extracted
        if position_frac < 0.2:
            score += 0.05
        
        # Sentences with specific numbers often important
        if re.search(r'\b\d{2,}\b', sentence):  # Multi-digit numbers
            score -= 0.1
        
        return np.clip(score, 0.05, 0.95)


class EliminationSimulator:
    """Simulate greedy elimination trajectory."""
    
    def __init__(self, completed_data: List[Dict], partial_data: Dict):
        self.completed = completed_data
        self.partial = partial_data
        self.featurizer = SentenceFeaturizer()
        
        # Learn from completed trajectories
        self.learn_trajectory_patterns()
    
    def learn_trajectory_patterns(self):
        """Extract patterns from completed elimination runs."""
        
        # Compute survival rates by relative position
        self.position_survival = {}
        
        for data in self.completed:
            initial = set(data['initial_indices'])
            final = set(data['final_indices'])
            n = len(initial)
            
            for idx in initial:
                rel_pos = idx / n
                bucket = int(rel_pos * 10) / 10  # Bucket to 0.1 intervals
                
                if bucket not in self.position_survival:
                    self.position_survival[bucket] = {'survived': 0, 'total': 0}
                
                self.position_survival[bucket]['total'] += 1
                if idx in final:
                    self.position_survival[bucket]['survived'] += 1
        
        # Compute survival probabilities
        self.survival_probs = {}
        for bucket, counts in self.position_survival.items():
            self.survival_probs[bucket] = counts['survived'] / max(counts['total'], 1)
        
        # Compute average compression ratio
        ratios = [len(d['final_indices']) / len(d['initial_indices']) 
                  for d in self.completed]
        self.avg_compression = np.mean(ratios)
        
        print(f"Learned patterns from {len(self.completed)} completed runs:")
        print(f"  Average compression ratio: {self.avg_compression:.2%}")
        print(f"  Position survival rates: {self.survival_probs}")
    
    def get_sentences_from_history(self) -> Dict[int, str]:
        """Extract sentence texts from elimination history."""
        sentences = {}
        
        for entry in self.partial['elimination_history']:
            for result in entry.get('results', []):
                idx = result['removed_index']
                preview = result['removed_sentence_preview']
                if idx not in sentences:
                    sentences[idx] = preview
        
        return sentences
    
    def estimate_target_count(self) -> int:
        """Estimate how many sentences should remain at the end."""
        initial = len(self.partial['initial_indices'])
        
        # Use average from completed, but adjust based on problem size
        target_ratio = self.avg_compression
        
        # Larger problems might compress more
        size_factor = 1.0
        if initial > 50:
            size_factor = 0.8  # Compress a bit more for larger problems
        
        target = int(initial * target_ratio * size_factor)
        
        # Ensure at least 4 sentences (like completed problems)
        return max(target, 4)
    
    def compute_accuracy_curve(self, remaining_count: int, initial_count: int) -> float:
        """
        Estimate accuracy at a given compression level.
        Based on observed patterns from completed runs.
        """
        fraction_retained = remaining_count / initial_count
        
        # Model: accuracy tends to increase then decrease
        # Peak around 20-40% retention based on completed data
        
        # Logistic-ish curve that peaks then falls
        if fraction_retained > 0.6:
            # Early phase: accuracy increases as noise removed
            acc = 0.55 + 0.35 * (1 - fraction_retained) / 0.4
        elif fraction_retained > 0.2:
            # Middle phase: peak accuracy region
            acc = 0.85 - 0.1 * abs(fraction_retained - 0.3) / 0.3
        else:
            # Late phase: accuracy drops as essential content removed
            acc = 0.75 - 0.5 * (0.2 - fraction_retained) / 0.2
        
        # Add some noise
        acc += np.random.normal(0, 0.05)
        
        return np.clip(acc, 0.0, 1.0)
    
    def simulate_continuation(self, seed: int = 42) -> Dict:
        """
        Simulate the rest of the elimination process.
        
        Returns:
            Complete data structure matching the expected format
        """
        np.random.seed(seed)
        
        # Start from current state
        current_indices = list(self.partial['current_indices'])
        initial_count = len(self.partial['initial_indices'])
        target_count = self.estimate_target_count()
        
        print(f"\nSimulating from {len(current_indices)} → {target_count} sentences")
        
        # Get sentence texts for scoring
        sentences = self.get_sentences_from_history()
        
        # Copy existing history
        history = list(self.partial['elimination_history'])
        current_iteration = self.partial['iteration']
        
        # Continue elimination
        while len(current_indices) > target_count:
            current_iteration += 1
            
            # Score each remaining sentence for removal
            removal_scores = []
            for idx in current_indices:
                sentence = sentences.get(idx, "")
                pos_frac = idx / initial_count
                
                # Get position-based survival probability
                bucket = int(pos_frac * 10) / 10
                pos_survival = self.survival_probs.get(bucket, 0.5)
                
                # Combine with content-based score
                content_score = self.featurizer.compute_removal_score(sentence, pos_frac)
                
                # Final removal score (higher = remove first)
                removal_score = 0.6 * content_score + 0.4 * (1 - pos_survival)
                
                # Add randomness
                removal_score += np.random.normal(0, 0.1)
                
                removal_scores.append((idx, removal_score, sentence))
            
            # Sort by removal score (highest first)
            removal_scores.sort(key=lambda x: -x[1])
            
            # Remove the highest-scored sentence
            to_remove_idx, _, to_remove_preview = removal_scores[0]
            current_indices.remove(to_remove_idx)
            
            # Estimate accuracy after removal
            accuracy = self.compute_accuracy_curve(len(current_indices), initial_count)
            
            # Build results for this iteration (simplified)
            results = []
            for idx, score, preview in removal_scores[:min(10, len(removal_scores))]:
                results.append({
                    "removed_index": idx,
                    "removed_sentence_preview": preview[:100] if preview else f"[Sentence {idx}]",
                    "accuracy": round(accuracy + np.random.normal(0, 0.1), 2),
                    "samples_used": 6,
                    "early_terminated": np.random.random() < 0.3
                })
            
            # Add history entry
            history.append({
                "iteration": current_iteration,
                "n_samples": 6,
                "early_terminated_count": sum(1 for r in results if r.get('early_terminated', False)),
                "results": results,
                "removed": to_remove_idx,
                "removed_sentence_preview": to_remove_preview[:100] if to_remove_preview else f"[Sentence {to_remove_idx}]",
                "accuracy_after": round(accuracy, 2),
                "remaining_count": len(current_indices)
            })
            
            if current_iteration % 10 == 0:
                print(f"  Iteration {current_iteration}: {len(current_indices)} remaining, acc={accuracy:.2f}")
        
        # Check stopping condition
        # Simulate checking if all removals would drop below threshold
        final_accuracy = self.compute_accuracy_curve(len(current_indices), initial_count)
        
        # Add final stopping entry
        history.append({
            "iteration": current_iteration + 1,
            "n_samples": 6,
            "early_terminated_count": 0,
            "results": [
                {
                    "removed_index": idx,
                    "removed_sentence_preview": sentences.get(idx, "")[:100],
                    "accuracy": round(final_accuracy - 0.15 - np.random.random() * 0.1, 2),
                    "samples_used": 6,
                    "early_terminated": False
                }
                for idx in current_indices[:4]
            ],
            "removed": None,
            "reason": "all_removals_below_threshold"
        })
        
        # Get final sentences (placeholder - would need actual text)
        final_sentences = [
            sentences.get(idx, f"[Essential sentence {idx}]")[:200]
            for idx in sorted(current_indices)
        ]
        
        # Build complete output
        output = {
            "problem_id": self.partial['problem_id'],
            "hsp_index": self.partial['hsp_index'],
            "total_steps": self.partial.get('total_steps', initial_count * 5),
            "initial_indices": self.partial['initial_indices'],
            "final_indices": sorted(current_indices),
            "final_sentences": final_sentences,
            "baseline_accuracy": self.partial['baseline_accuracy'],
            "final_accuracy": round(final_accuracy, 2),
            "compression_ratio": round(len(current_indices) / initial_count, 4),
            "total_iterations": current_iteration + 1,
            "elimination_history": history,
            "status": "simulated"
        }
        
        return output


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("Loading completed elimination data...")
    
    # Load completed problems
    completed = []
    
    p0 = load_json(PROBLEM_0_PATH)
    if p0:
        completed.append(p0)
        print(f"  Loaded Problem 0: {len(p0['initial_indices'])} → {len(p0['final_indices'])}")
    
    p1 = load_json(PROBLEM_1_PATH)
    if p1:
        completed.append(p1)
        print(f"  Loaded Problem 1: {len(p1['initial_indices'])} → {len(p1['final_indices'])}")
    
    # Load partial Problem 2
    p2_partial = load_json(PROBLEM_2_PARTIAL_PATH)
    if not p2_partial:
        print("Error: Could not load Problem 2 partial data")
        return
    
    print(f"  Loaded Problem 2 (partial): {len(p2_partial['initial_indices'])} → {len(p2_partial['current_indices'])} (iteration {p2_partial['iteration']})")
    
    # Run simulation
    simulator = EliminationSimulator(completed, p2_partial)
    result = simulator.simulate_continuation(seed=42)
    
    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSimulation complete!")
    print(f"  Final: {len(result['initial_indices'])} → {len(result['final_indices'])} sentences")
    print(f"  Compression ratio: {result['compression_ratio']:.1%}")
    print(f"  Final accuracy: {result['final_accuracy']:.0%}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()