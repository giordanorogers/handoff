import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
INPUT_FILE = "experiment_1_results.jsonl"
OUTPUT_DIR = "plots"
SAMPLE_SIZE = 15  # Number of individual curves to plot in Figure 1

import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def plot_individual_curves(data):
    """
    Figure 1: Plots k (step index) vs P(correct) for a subset of problems.
    Shows the distinct 'step functions' of reasoning.
    """
    plt.figure(figsize=(10, 6))
    
    # Filter for problems that actually found an ESSP to make the plot cleaner
    solved_data = [d for d in data if d['essp_index'] != -1]
    
    # Take a random sample or first N
    subset = solved_data[:SAMPLE_SIZE]
    
    for i, datum in enumerate(subset):
        accs = datum['step_accuracies']
        steps = range(len(accs))
        
        # Plot line
        plt.plot(steps, accs, alpha=0.6, linewidth=2, label=f"Prob {i}")
        
        # Mark ESSP with a star
        essp_idx = datum['essp_index']
        if essp_idx < len(accs):
            plt.plot(essp_idx, accs[essp_idx], 'r*', markersize=10, zorder=10)

    plt.axhline(y=0.5, color='k', linestyle='--', label='Threshold (tau=0.5)')
    plt.xlabel("Reasoning Step (k)")
    plt.ylabel("P(correct | k)")
    plt.title(f"Sufficiency Curves (First {SAMPLE_SIZE} Solved Problems)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_1_individual_curves.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/figure_1_individual_curves.png")

def plot_aggregate_curve(data):
    """
    Figure 2: Normalized Progress (0-100%) vs Average P(correct).
    Since problems have different lengths, we normalize x-axis to % of CoT completed.
    """
    plt.figure(figsize=(10, 6))
    
    normalized_x = np.linspace(0, 1, 100) # Standardize to 100 points
    interpolated_y = []

    count = 0
    for datum in data:
        accs = datum['step_accuracies']
        if datum['essp_index'] == -1: continue # Skip non-converging CoTs
        if len(accs) < 2: continue # Skip single-step weirdness
        count += 1
        # Create x-axis for this specific problem (0.0 to 1.0)
        x_original = np.linspace(0, 1, len(accs))
        
        # Interpolate this problem's accuracy onto the standard grid
        f = interp1d(x_original, accs, kind='linear', bounds_error=False, fill_value=(accs[0], accs[-1]))
        y_interp = f(normalized_x)
        interpolated_y.append(y_interp)

    # Calculate Mean and Standard Deviation
    y_matrix = np.array(interpolated_y)
    y_mean = np.mean(y_matrix, axis=0)
    y_std = np.std(y_matrix, axis=0)
    
    # Plot Mean
    plt.plot(normalized_x * 100, y_mean, color='blue', linewidth=3, label='Mean Sufficiency')
    
    # Plot Confidence Interval
    plt.fill_between(normalized_x * 100, 
                     np.maximum(0, y_mean - y_std), 
                     np.minimum(1, y_mean + y_std), 
                     color='blue', alpha=0.2, label='Std Dev')

    plt.axhline(y=0.5, color='r', linestyle='--', label='Success Threshold')
    plt.xlabel("Reasoning Progress (%)")
    plt.ylabel("Average Probability of Correct Answer")
    plt.title(f"Aggregate Sufficiency Curve (N={count})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_2_aggregate_curve.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/figure_2_aggregate_curve.png")

def plot_essp_histogram(data):
    """
    Figure 3: Distribution of Relative ESSP Positions.
    Answers: "Does the model usually find the answer at 10% or 90% of the CoT?"
    """
    plt.figure(figsize=(8, 6))
    
    ratios = []
    failed_count = 0
    
    for datum in data:
        if datum['essp_index'] == -1:
            failed_count += 1
        else:
            # ESSP Index / Total Steps
            ratios.append(datum['essp_index'] / datum['total_steps'])
            
    plt.hist(ratios, bins=20, color='green', alpha=0.7, edgecolor='black')
    
    plt.xlabel("Relative Position of ESSP (0.0 = Start, 1.0 = End)")
    plt.ylabel("Number of Problems")
    plt.title(f"Distribution of Sufficiency Points\n(Solved: {len(ratios)}, Failed: {failed_count})")
    plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_3_essp_dist.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/figure_3_essp_dist.png")

if __name__ == "__main__":
    print("Loading data...")
    all_data = load_data(INPUT_FILE)
    print(f"Loaded {len(all_data)} records.")
    
    if not all_data:
        print("No data found! Check file path.")
    else:
        plot_individual_curves(all_data)
        plot_aggregate_curve(all_data)
        plot_essp_histogram(all_data)
        print("Done.")