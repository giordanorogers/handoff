import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from _src.analysis_utils import aggregate_results

def plot_hsr_comparison(df: pd.DataFrame, output_path: str):
    """
    Generates the validation plot: HSR vs. Trace Progress.
    """
    plt.figure(figsize=(10, 6))
    
    # Set theme
    sns.set_theme(style="whitegrid")
    
    # Plot Coherent
    coh = df[df['condition'] == 'Coherent']
    plt.plot(coh['progress'], coh['hsr_mean'], label='Correct Trace', color='#2ca02c', linewidth=2.5)
    plt.fill_between(coh['progress'], 
                     coh['hsr_mean'] - coh['hsr_std']*0.2, # Narrower band for visibility 
                     coh['hsr_mean'] + coh['hsr_std']*0.2, 
                     color='#2ca02c', alpha=0.15)

    # Plot Incoherent
    inc = df[df['condition'] == 'Shuffled (Incoherent)']
    plt.plot(inc['progress'], inc['hsr_mean'], label='Shuffled Trace', color='#d62728', linewidth=2.5, linestyle='--')
    plt.fill_between(inc['progress'], 
                     inc['hsr_mean'] - inc['hsr_std']*0.2, 
                     inc['hsr_mean'] + inc['hsr_std']*0.2, 
                     color='#d62728', alpha=0.15)

    plt.xlabel('Trace Progress (%)', fontsize=12)
    plt.ylabel('Handoff Success Rate (HSR)', fontsize=12)
    plt.title('HSR Validation: Coherent vs. Incoherent Reasoning', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xlim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main(args):
    # 1. Load Results
    result_path = os.path.join(args.input_dir, "results.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Results file not found at {result_path}")
        
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    print(f"Analyzing {len(data)} traces...")

    # 2. Aggregate Data
    df = aggregate_results(data)
    
    # 3. Calculate Summary Stats
    coh_final = df[(df['condition'] == 'Coherent') & (df['progress'] == 100)]['hsr_mean'].values[0]
    inc_final = df[(df['condition'] == 'Shuffled (Incoherent)') & (df['progress'] == 100)]['hsr_mean'].values[0]
    
    print("\n--- Experiment 1 Summary ---")
    print(f"Final HSR (Coherent):   {coh_final:.2%}")
    print(f"Final HSR (Shuffled):   {inc_final:.2%}")
    print(f"Validation Delta:       {coh_final - inc_final:.2%} (Should be large positive)")
    
    # 4. Generate Plot
    output_plot = os.path.join(args.input_dir, "exp1_validation_plot.png")
    plot_hsr_comparison(df, output_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/exp1_sanity")
    args = parser.parse_args()
    main(args)