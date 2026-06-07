import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths - adjust as needed
OUTPUT_DIR = Path("data/plots/attribution_plots")

CONFIG = {
    # Colors
    "critical_color": "#009957",      # Green for transfer-critical
    "non_critical_color": "#B0B0B0",  # Gray for non-critical
    "ig_color": "#4A90E2",
    "attn_color": "#E09000",
    
    # Line/bar styles
    "bar_width": 0.8,
    "line_width": 1.5,
    
    # Font settings
    "font_family": "serif",
    "font_size": 10,
    "small_font_size": 8,
    
    # Axes
    "axes_line_width": 1.0,
    "xtick_major_width": 0.5,
    "ytick_major_width": 0.5,
    "xtick_direction": "out",
    "ytick_direction": "out",
    
    # Grid
    "use_grid": True,
    "grid_alpha": 0.2,
    
    # Legend
    "legend_border_pad": 1.0,
    "legend_frame_alpha": 1.0,
    
    # Figure
    "figsize_single": (7, 3),
    "figsize_combined": (7, 6),
}

# Data (embedded for convenience - can also load from files)
PROBLEM_0_CRITICAL = [17, 27, 32, 37]
PROBLEM_1_CRITICAL = [5, 11, 13, 15]

PROBLEM_0_IG = {"0": 0.0439, "1": 0.0236, "2": 0.0215, "3": 0.0150, "4": 0.0244, "5": 0.0148, "6": 0.0107, "7": 0.0173, "8": 0.0301, "9": 0.0413, "10": 0.0075, "11": 0.0234, "12": 0.0371, "13": 0.0395, "14": 0.0232, "15": 0.0154, "16": 0.0109, "17": 0.0138, "18": 0.0831, "19": 0.0253, "20": 0.0236, "21": 0.0179, "22": 0.0211, "23": 0.0209, "24": 0.0203, "25": 0.0066, "26": 0.0248, "27": 0.0073, "28": 0.0247, "29": 0.0407, "30": 0.0616, "31": 0.0347, "32": 0.0386, "33": 0.0332, "34": 0.0464, "35": 0.0330, "36": 0.0091, "37": 0.0075, "38": 0.0063}
PROBLEM_1_IG = {"0": 0.265, "1": 0.083, "2": 0.128, "3": 0.159, "4": 0.122, "5": 0.031, "6": 0.031, "7": 0.029, "8": 0.022, "9": 0.036, "10": 0.035, "11": 0.022, "12": 0.022, "13": 0.007, "14": 0.004, "15": 0.006}

PROBLEM_0_ATTN = {"0": 0.127, "1": 3.43e-21, "2": 8.99e-09, "3": 0.125, "4": 0.136, "5": 2.86e-25, "6": 7.29e-30, "7": 2.31e-27, "8": 0.002, "9": 5.80e-13, "10": 2.86e-20, "11": 3.07e-38, "12": 8.50e-24, "13": 4.46e-16, "14": 9.53e-27, "15": 3.79e-25, "16": 4.37e-20, "17": 5.36e-13, "18": 2.42e-10, "19": 2.50e-22, "20": 6.33e-23, "21": 3.04e-31, "22": 2.72e-17, "23": 8.71e-19, "24": 1.06e-20, "25": 3.14e-32, "26": 2.99e-15, "27": 1.06e-28, "28": 5.30e-20, "29": 6.79e-21, "30": 7.15e-16, "31": 2.72e-19, "32": 1.06e-15, "33": 1.14e-24, "34": 1.90e-17, "35": 1.91e-27, "36": 6.48e-25, "37": 5.54e-27, "38": 4.76e-19}
PROBLEM_1_ATTN = {"0": 9.25e-19, "1": 0.292, "2": 3.50e-21, "3": 3.43e-16, "4": 8.53e-24, "5": 3.60e-32, "6": 3.04e-19, "7": 4.66e-22, "8": 2.61e-31, "9": 1.28e-30, "10": 9.76e-18, "11": 2.46e-36, "12": 8.35e-35, "13": 1.57e-43, "14": 7.86e-44, "15": 1e-50}


class AttributionPlotter:
    
    def __init__(self, config):
        self.config = config
        
    def set_matplotlib_parameters(self):
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['font.size'] = self.config['font_size']
        plt.rcParams['axes.linewidth'] = self.config['axes_line_width']
        plt.rcParams['xtick.major.width'] = self.config['xtick_major_width']
        plt.rcParams['ytick.major.width'] = self.config['ytick_major_width']
        plt.rcParams['xtick.direction'] = self.config['xtick_direction']
        plt.rcParams['ytick.direction'] = self.config['ytick_direction']
    
    def get_ranked_data(self, scores_dict, critical_set):
        """
        Sort sentences by score (descending) and mark critical ones.
        
        Returns:
            List of dicts with idx, score, is_critical, rank
        """
        entries = []
        for idx_str, score in scores_dict.items():
            idx = int(idx_str)
            entries.append({
                'idx': idx,
                'score': score,
                'is_critical': idx in critical_set,
            })
        
        # Sort by score descending
        entries.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for rank, entry in enumerate(entries):
            entry['rank'] = rank + 1
        
        return entries
    
    def compute_precision_at_k(self, ranked_data, critical_set, k):
        """Compute precision@k."""
        top_k_indices = [e['idx'] for e in ranked_data[:k]]
        hits = sum(1 for idx in top_k_indices if idx in critical_set)
        return hits / min(k, len(critical_set))
    
    def plot_attribution_bars(self, scores_dict, critical_set, title, ax, 
                               ylabel='Attribution Score', show_legend=True,
                               show_xticklabels=True):
        """
        Plot bar chart of attribution scores sorted by score.
        Critical sentences are highlighted in green.
        """
        ranked = self.get_ranked_data(scores_dict, critical_set)
        
        x_positions = np.arange(len(ranked))
        scores = [e['score'] for e in ranked]
        colors = [
            self.config['critical_color'] if e['is_critical'] 
            else self.config['non_critical_color'] 
            for e in ranked
        ]
        labels = [f"S{e['idx']}" for e in ranked]
        
        bars = ax.bar(
            x_positions, 
            scores, 
            color=colors,
            width=self.config['bar_width'],
        )
        
        ax.set_xticks(x_positions)
        if show_xticklabels:
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=self.config['small_font_size'])
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if self.config['use_grid']:
            ax.grid(visible=True, alpha=self.config['grid_alpha'], axis='y')
        
        # Add legend
        if show_legend:
            critical_patch = mpatches.Patch(
                color=self.config['critical_color'], 
                label='Transfer-critical'
            )
            non_critical_patch = mpatches.Patch(
                color=self.config['non_critical_color'], 
                label='Non-critical'
            )
            ax.legend(
                handles=[critical_patch, non_critical_patch],
                loc='upper right',
                fontsize=self.config['small_font_size'],
            )
        
        return ranked
    
    def plot_combined(self, output_path):
        """Create combined figure with both problems and both methods (2x2 grid)."""
        self.set_matplotlib_parameters()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        
        # Row 0: Problem 0 (IG and Attention)
        ranked_0_ig = self.plot_attribution_bars(
            PROBLEM_0_IG,
            PROBLEM_0_CRITICAL,
            'Problem 0: Integrated Gradients',
            axes[0, 0],
            ylabel='IG Score',
            show_legend=True,
            show_xticklabels=True,
        )
        
        ranked_0_attn = self.plot_attribution_bars(
            PROBLEM_0_ATTN,
            PROBLEM_0_CRITICAL,
            'Problem 0: Attention',
            axes[0, 1],
            ylabel='Attention Score',
            show_legend=False,
            show_xticklabels=True,
        )
        
        # Row 1: Problem 1 (IG and Attention)
        ranked_1_ig = self.plot_attribution_bars(
            PROBLEM_1_IG,
            PROBLEM_1_CRITICAL,
            'Problem 1: Integrated Gradients',
            axes[1, 0],
            ylabel='IG Score',
            show_legend=False,
            show_xticklabels=True,
        )
        
        ranked_1_attn = self.plot_attribution_bars(
            PROBLEM_1_ATTN,
            PROBLEM_1_CRITICAL,
            'Problem 1: Attention',
            axes[1, 1],
            ylabel='Attention Score',
            show_legend=False,
            show_xticklabels=True,
        )
        
        # Add shared x-axis label at bottom
        fig.text(0.5, 0.02, 'Sentence Index (sorted by attribution score, descending)', 
                 ha='center', va='center', fontsize=self.config['font_size'])
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for the shared label
        
        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.pdf'))
        print(f"Saved to {output_path}")
        plt.close()
        
        # Print statistics
        print("\n" + "="*60)
        print("INTEGRATED GRADIENTS")
        print("="*60)
        self.print_statistics(ranked_0_ig, PROBLEM_0_CRITICAL, "Problem 0 (IG)")
        self.print_statistics(ranked_1_ig, PROBLEM_1_CRITICAL, "Problem 1 (IG)")
        
        print("\n" + "="*60)
        print("ATTENTION")
        print("="*60)
        self.print_statistics(ranked_0_attn, PROBLEM_0_CRITICAL, "Problem 0 (Attn)")
        self.print_statistics(ranked_1_attn, PROBLEM_1_CRITICAL, "Problem 1 (Attn)")
    
    def print_statistics(self, ranked_data, critical_set, name):
        """Print precision@k and rank statistics."""
        print(f"\n{name}:")
        print(f"  Critical set: {sorted(critical_set)}")
        
        # Precision@k
        for k in [4, 10]:
            if k <= len(ranked_data):
                p_at_k = self.compute_precision_at_k(ranked_data, critical_set, k)
                print(f"  Precision@{k}: {p_at_k:.0%}")
        
        # Random baseline
        baseline = len(critical_set) / len(ranked_data)
        print(f"  Random baseline: {baseline:.1%}")
        
        # Ranks of critical sentences
        critical_ranks = [e for e in ranked_data if e['is_critical']]
        rank_strs = [f"S{e['idx']}→#{e['rank']}" for e in critical_ranks]
        print(f"  Critical sentence ranks: {', '.join(rank_strs)}")
        
        avg_rank = np.mean([e['rank'] for e in critical_ranks])
        print(f"  Average rank of critical: {avg_rank:.1f} / {len(ranked_data)}")


if __name__ == "__main__":
    plotter = AttributionPlotter(CONFIG)
    plotter.plot_combined(OUTPUT_DIR / "attribution_comparison.png")