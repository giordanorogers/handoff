import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/plots/essp_hsp_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "essp_color": "#5344C6",
    "hsp_color": "#009957",
    "font_family": "serif",
    "font_size": 11,
    "axes_line_width": 1.0,
    "xtick_major_width": 1.0,
    "ytick_major_width": 1.0,
    "xtick_direction": "out",
    "ytick_direction": "out",
}

plt.rcParams['font.family'] = CONFIG['font_family']
plt.rcParams['font.size'] = CONFIG['font_size']
plt.rcParams['axes.linewidth'] = CONFIG['axes_line_width']
plt.rcParams['xtick.major.width'] = CONFIG['xtick_major_width']
plt.rcParams['ytick.major.width'] = CONFIG['ytick_major_width']
plt.rcParams['xtick.direction'] = CONFIG['xtick_direction']
plt.rcParams['ytick.direction'] = CONFIG['ytick_direction']

# Load data
dataset = [
    {  # Problem 1
        "original_essp_accuracy": 1.0,
        "pruned_essp_accuracy": 0.0,
        "anti_pruned_essp_accuracy": 1.0,
        "anti_pruned_essp_accuracy_random_sample": 
        "essp_retention": 0.38,  # 27/72 sentences retained
        "original_hsp_accuracy": 0.6,
        "pruned_hsp_accuracy": 0.45,
        "ant-pruned_hsp_accuracy": 0.3,
        "hsp_retention": 0.43,  # 17/39 sentences retained
    }
    # Add more problems here as you collect data
]

def compute_averages(dataset):
    """Compute average accuracies across all problems."""
    n = len(dataset)
    return {
        "original_essp": np.mean([d["original_essp_accuracy"] for d in dataset]),
        "pruned_essp": np.mean([d["pruned_essp_accuracy"] for d in dataset]),
        "original_hsp": np.mean([d["original_hsp_accuracy"] for d in dataset]),
        "pruned_hsp": np.mean([d["pruned_hsp_accuracy"] for d in dataset]),
        "essp_retention": np.mean([d["essp_retention"] for d in dataset]),
        "hsp_retention": np.mean([d["hsp_retention"] for d in dataset]),
    }

def plot_pruning_comparison(dataset, output_path=None):
    """Create grouped bar chart comparing full vs pruned accuracy."""
    avg = compute_averages(dataset)
    n_problems = len(dataset)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.array([0, 1])
    width = 0.35
    
    # Bar positions
    essp_positions = x - width/2
    hsp_positions = x + width/2
    
    # ESSP bars
    essp_vals = [avg["original_essp"], avg["pruned_essp"]]
    bars_essp = ax.bar(essp_positions, essp_vals, width, 
                       label='Early-Stopping (ESSP)', 
                       color=CONFIG["essp_color"], alpha=0.85)
    
    # HSP bars
    hsp_vals = [avg["original_hsp"], avg["pruned_hsp"]]
    bars_hsp = ax.bar(hsp_positions, hsp_vals, width, 
                      label='Handoff (HSP)', 
                      color=CONFIG["hsp_color"], alpha=0.85)
    
    # Add value labels on bars
    for bar, val in zip(bars_essp, essp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    for bar, val in zip(bars_hsp, hsp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Add retention annotation
    retention_text = (f"Pruned traces retain ~{avg['essp_retention']*100:.0f}% (ESSP) "
                      f"and ~{avg['hsp_retention']*100:.0f}% (HSP) of sentences")
    ax.text(0.5, -0.15, retention_text, transform=ax.transAxes, 
            ha='center', fontsize=10, style='italic')
    
    # Formatting
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Full Prefix', 'Pruned Prefix'])
    ax.legend(loc='upper right')
    ax.set_title(f'Accuracy Before and After Gradient-Based Pruning (N={n_problems})')
    
    # Add horizontal line at 0.5 threshold
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(1.02, 0.5, 'threshold', transform=ax.get_yaxis_transform(), 
            va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    plt.show()
    return fig, ax

if __name__ == "__main__":
    plot_pruning_comparison(dataset, OUTPUT_DIR / "pruning_accuracy_comparison.png")