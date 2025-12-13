import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data (replace with your actual file path)
# For this example, I'll parse the truncated data you provided
data = [
    {
        "id": 0,
        "hsr_coherent": [{"step_idx": i, "hsr": 1.0} for i in range(150)] + 
                        [{"step_idx": 231, "hsr": 0.0}, {"step_idx": 232, "hsr": 0.8}],
        "hsr_incoherent": [{"step_idx": i, "hsr": np.random.choice([0.8, 1.0], p=[0.1, 0.9])} 
                          for i in range(233)]
    },
    {
        "id": 1,
        "hsr_coherent": [{"step_idx": i, "hsr": np.random.choice([0.8, 1.0], p=[0.15, 0.85])} 
                        for i in range(123)],
        "hsr_incoherent": [{"step_idx": i, "hsr": np.random.choice([0.8, 1.0], p=[0.2, 0.8])} 
                          for i in range(123)]
    }
]

def load_data(filepath):
    """Load HSR data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_hsr_values(problem_data, key='hsr_coherent'):
    """Extract step indices and HSR values from a problem."""
    steps = problem_data[key]
    indices = [s['step_idx'] for s in steps]
    hsrs = [s['hsr'] for s in steps]
    return indices, hsrs

def plot_single_problem(problem_data, ax=None, show_legend=True):
    """Plot HSR for coherent vs incoherent for a single problem."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    # Extract data
    coh_idx, coh_hsr = extract_hsr_values(problem_data, 'hsr_coherent')
    incoh_idx, incoh_hsr = extract_hsr_values(problem_data, 'hsr_incoherent')
    
    # Plot
    ax.plot(coh_idx, coh_hsr, 'b-', alpha=0.7, linewidth=1.5, label='Coherent')
    ax.plot(incoh_idx, incoh_hsr, 'r-', alpha=0.7, linewidth=1.5, label='Incoherent')
    
    # Highlight drops below 1.0
    coh_drops = [(i, h) for i, h in zip(coh_idx, coh_hsr) if h < 1.0]
    incoh_drops = [(i, h) for i, h in zip(incoh_idx, incoh_hsr) if h < 1.0]
    
    if coh_drops:
        ax.scatter(*zip(*coh_drops), c='blue', s=30, zorder=5, marker='v')
    if incoh_drops:
        ax.scatter(*zip(*incoh_drops), c='red', s=30, zorder=5, marker='v')
    
    ax.set_xlabel('Step Index')
    ax.set_ylabel('HSR')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"Problem {problem_data['id']}: Handoff Success Rate")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='lower left')
    
    return ax

def plot_hsr_comparison(data, ncols=2):
    """Plot HSR for all problems in a grid."""
    n_problems = len(data)
    nrows = (n_problems + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4*nrows))
    axes = axes.flatten() if n_problems > 1 else [axes]
    
    for i, problem in enumerate(data):
        plot_single_problem(problem, ax=axes[i], show_legend=(i == 0))
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def compute_summary_stats(data):
    """Compute summary statistics for all problems."""
    stats = []
    for problem in data:
        coh_idx, coh_hsr = extract_hsr_values(problem, 'hsr_coherent')
        incoh_idx, incoh_hsr = extract_hsr_values(problem, 'hsr_incoherent')
        
        stats.append({
            'id': problem['id'],
            'coherent_mean': np.mean(coh_hsr),
            'coherent_min': np.min(coh_hsr),
            'coherent_drops': sum(1 for h in coh_hsr if h < 1.0),
            'coherent_steps': len(coh_hsr),
            'incoherent_mean': np.mean(incoh_hsr),
            'incoherent_min': np.min(incoh_hsr),
            'incoherent_drops': sum(1 for h in incoh_hsr if h < 1.0),
            'incoherent_steps': len(incoh_hsr),
        })
    return stats

def plot_aggregate_hsr(data):
    """Plot aggregate HSR statistics across all problems."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stats = compute_summary_stats(data)
    ids = [s['id'] for s in stats]
    
    # Mean HSR comparison
    ax1 = axes[0]
    x = np.arange(len(ids))
    width = 0.35
    ax1.bar(x - width/2, [s['coherent_mean'] for s in stats], width, label='Coherent', color='steelblue')
    ax1.bar(x + width/2, [s['incoherent_mean'] for s in stats], width, label='Incoherent', color='indianred')
    ax1.set_xlabel('Problem ID')
    ax1.set_ylabel('Mean HSR')
    ax1.set_title('Mean HSR by Problem')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ids)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Number of drops
    ax2 = axes[1]
    ax2.bar(x - width/2, [s['coherent_drops'] for s in stats], width, label='Coherent', color='steelblue')
    ax2.bar(x + width/2, [s['incoherent_drops'] for s in stats], width, label='Incoherent', color='indianred')
    ax2.set_xlabel('Problem ID')
    ax2.set_ylabel('Number of HSR Drops (<1.0)')
    ax2.set_title('HSR Drops by Problem')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids)
    ax2.legend()
    
    # Drop rate (normalized)
    ax3 = axes[2]
    coh_rates = [s['coherent_drops']/s['coherent_steps'] for s in stats]
    incoh_rates = [s['incoherent_drops']/s['incoherent_steps'] for s in stats]
    ax3.bar(x - width/2, coh_rates, width, label='Coherent', color='steelblue')
    ax3.bar(x + width/2, incoh_rates, width, label='Incoherent', color='indianred')
    ax3.set_xlabel('Problem ID')
    ax3.set_ylabel('Drop Rate')
    ax3.set_title('HSR Drop Rate (Drops/Total Steps)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ids)
    ax3.legend()
    
    plt.tight_layout()
    return fig

def plot_hsr_distribution(data):
    """Plot distribution of HSR values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_coh_hsr = []
    all_incoh_hsr = []
    
    for problem in data:
        _, coh_hsr = extract_hsr_values(problem, 'hsr_coherent')
        _, incoh_hsr = extract_hsr_values(problem, 'hsr_incoherent')
        all_coh_hsr.extend(coh_hsr)
        all_incoh_hsr.extend(incoh_hsr)
    
    # Histograms
    bins = np.linspace(0, 1.05, 22)
    axes[0].hist(all_coh_hsr, bins=bins, alpha=0.7, label='Coherent', color='steelblue', edgecolor='black')
    axes[0].hist(all_incoh_hsr, bins=bins, alpha=0.7, label='Incoherent', color='indianred', edgecolor='black')
    axes[0].set_xlabel('HSR Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of HSR Values')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot([all_coh_hsr, all_incoh_hsr], labels=['Coherent', 'Incoherent'])
    axes[1].set_ylabel('HSR Value')
    axes[1].set_title('HSR Distribution Comparison')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # To use with your actual data file:
    data = load_data('_results/exp1_sanity/results.json')
    
    # For demo, using synthetic data based on your structure
    print("Generating HSR visualizations...")
    
    # Plot individual problems
    fig1 = plot_hsr_comparison(data)
    fig1.savefig('hsr_by_problem.png', dpi=150, bbox_inches='tight')
    print("Saved: hsr_by_problem.png")
    
    # Plot aggregate statistics
    fig2 = plot_aggregate_hsr(data)
    fig2.savefig('hsr_aggregate.png', dpi=150, bbox_inches='tight')
    print("Saved: hsr_aggregate.png")
    
    # Plot distributions
    fig3 = plot_hsr_distribution(data)
    fig3.savefig('hsr_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: hsr_distribution.png")
    
    # Print summary
    print("\n=== Summary Statistics ===")
    stats = compute_summary_stats(data)
    for s in stats:
        print(f"\nProblem {s['id']}:")
        print(f"  Coherent:   mean={s['coherent_mean']:.3f}, min={s['coherent_min']:.2f}, drops={s['coherent_drops']}/{s['coherent_steps']}")
        print(f"  Incoherent: mean={s['incoherent_mean']:.3f}, min={s['incoherent_min']:.2f}, drops={s['incoherent_drops']}/{s['incoherent_steps']}")
    
    plt.show()