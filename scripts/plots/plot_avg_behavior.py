import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# ==== CONFIGURATION ====
DATA_DIR = './scripts/plots/'  # Current directory, change if needed
CSV_PATTERN = 'reasoning_steps_problem*.csv'  # Pattern to match problem files

# Define categories and colors
categories = ['planning', 'exploration', 'reasoning', 'insight', 'monitoring', 'correction']
colors = {
    'planning': '#3498db',
    'exploration': '#2ecc71',
    'reasoning': '#e74c3c',
    'insight': '#f39c12',
    'monitoring': '#9b59b6',
    'correction': '#95a5a6'
}

# ==== LOAD ALL PROBLEM FILES ====
csv_files = glob.glob(f'{DATA_DIR}/{CSV_PATTERN}')
print(f"Found {len(csv_files)} problem files:")
for f in csv_files:
    print(f"  - {f}")

if len(csv_files) == 0:
    print(f"\nERROR: No files matching pattern '{CSV_PATTERN}' found in {DATA_DIR}")
    exit(1)

# ==== COMPUTE PROPORTIONS FOR EACH PROBLEM ====
def get_proportions(subset):
    """Calculate proportion of each category in a subset."""
    counts = subset['behavioral_category'].value_counts()
    total = len(subset)
    return {cat: counts.get(cat, 0) / total for cat in categories}

all_handoff_props = []
all_early_stop_props = []
problem_info = []

for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    
    # Get problem identifier
    problem_name = Path(csv_file).stem
    
    # Split data
    handoff = df[df['is_handoff_sufficient'] == True]
    early_stop_all = df[df['is_early_stop_sufficient'] == True]
    
    # Calculate proportions
    handoff_props = get_proportions(handoff)
    early_stop_props = get_proportions(early_stop_all)
    
    all_handoff_props.append(handoff_props)
    all_early_stop_props.append(early_stop_props)
    
    # Store metadata
    handoff_boundary = handoff['step_number'].max()
    early_stop_boundary = early_stop_all['step_number'].max()
    problem_info.append({
        'name': problem_name,
        'handoff_boundary': handoff_boundary,
        'early_stop_boundary': early_stop_boundary,
        'total_steps': len(df)
    })
    
    print(f"\n{problem_name}:")
    print(f"  Handoff boundary: step {handoff_boundary}")
    print(f"  Early-stop boundary: step {early_stop_boundary}")
    print(f"  Total steps: {len(df)}")

# ==== COMPUTE AVERAGES ====
def average_proportions(props_list):
    """Average proportions across multiple problems."""
    avg_props = {}
    for cat in categories:
        values = [props[cat] for props in props_list]
        avg_props[cat] = np.mean(values)
    return avg_props

avg_handoff_props = average_proportions(all_handoff_props)
avg_early_stop_props = average_proportions(all_early_stop_props)

# ==== COMPUTE STANDARD DEVIATIONS ====
def std_proportions(props_list):
    """Standard deviation of proportions across multiple problems."""
    std_props = {}
    for cat in categories:
        values = [props[cat] for props in props_list]
        std_props[cat] = np.std(values)
    return std_props

std_handoff_props = std_proportions(all_handoff_props)
std_early_stop_props = std_proportions(all_early_stop_props)

# ==== FIGURE 1: AVERAGED DISTRIBUTION ====
fig1, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(categories))
width = 0.35

# Plot bars with error bars
bars1 = ax.bar(x - width/2, [avg_handoff_props[cat] for cat in categories], 
               width, label=f'Handoff-sufficient (avg over {len(csv_files)} problems)',
               color=[colors[cat] for cat in categories], alpha=0.8, 
               edgecolor='black', linewidth=0.5,
               yerr=[std_handoff_props[cat] for cat in categories],
               capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

bars2 = ax.bar(x + width/2, [avg_early_stop_props[cat] for cat in categories], 
               width, label=f'Early-stop sufficient (avg over {len(csv_files)} problems)',
               color=[colors[cat] for cat in categories], alpha=0.4, 
               edgecolor='black', linewidth=0.5,
               yerr=[std_early_stop_props[cat] for cat in categories],
               capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

ax.set_ylabel('Proportion of Steps', fontsize=12)
ax.set_xlabel('Behavioral Category', fontsize=12)
ax.set_title(f'Average Distribution of Behavioral Categories (N={len(csv_files)} problems)', 
             fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
ax.legend(frameon=True, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('average_category_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==== FIGURE 2: HEATMAP SHOWING VARIATION ACROSS PROBLEMS ====
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for heatmap
handoff_matrix = np.array([[props[cat] for cat in categories] for props in all_handoff_props])
early_stop_matrix = np.array([[props[cat] for cat in categories] for props in all_early_stop_props])

problem_labels = [info['name'].replace('reasoning_steps_', 'P') for info in problem_info]

# Handoff heatmap
im1 = ax1.imshow(handoff_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
ax1.set_yticks(range(len(problem_labels)))
ax1.set_yticklabels(problem_labels)
ax1.set_title('Handoff-sufficient Region', fontsize=12, fontweight='bold')
ax1.set_ylabel('Problem', fontsize=11)

# Add text annotations
for i in range(len(problem_labels)):
    for j in range(len(categories)):
        text = ax1.text(j, i, f'{handoff_matrix[i, j]*100:.1f}%',
                       ha="center", va="center", color="black", fontsize=9)

# Early-stop heatmap
im2 = ax2.imshow(early_stop_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
ax2.set_yticks(range(len(problem_labels)))
ax2.set_yticklabels(problem_labels)
ax2.set_title('Early-stop Sufficient Region', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(problem_labels)):
    for j in range(len(categories)):
        text = ax2.text(j, i, f'{early_stop_matrix[i, j]*100:.1f}%',
                       ha="center", va="center", color="black", fontsize=9)

# Add colorbars
fig2.colorbar(im1, ax=ax1, label='Proportion')
fig2.colorbar(im2, ax=ax2, label='Proportion')

plt.suptitle('Category Distribution Across Problems', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('problems_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ==== PRINT SUMMARY STATISTICS ====
print("\n" + "="*70)
print(f"AVERAGED STATISTICS OVER {len(csv_files)} PROBLEMS")
print("="*70)

print("\n--- HANDOFF-SUFFICIENT REGION ---")
for cat in categories:
    avg = avg_handoff_props[cat] * 100
    std = std_handoff_props[cat] * 100
    print(f"{cat.capitalize():15s}: {avg:5.1f}% ± {std:4.1f}%")

print("\n--- EARLY-STOP SUFFICIENT REGION ---")
for cat in categories:
    avg = avg_early_stop_props[cat] * 100
    std = std_early_stop_props[cat] * 100
    print(f"{cat.capitalize():15s}: {avg:5.1f}% ± {std:4.1f}%")

print("\n--- PROPORTIONAL DIFFERENCES (Handoff - Early-stop) ---")
print("Positive = enriched in handoff, Negative = enriched in full trace")
for cat in categories:
    diff = (avg_handoff_props[cat] - avg_early_stop_props[cat]) * 100
    if abs(diff) > 1:  # Only show differences > 1%
        direction = "↑" if diff > 0 else "↓"
        print(f"{cat.capitalize():15s}: {direction} {abs(diff):5.1f}%")

# ==== EXPORT DETAILED RESULTS ====
results_df = pd.DataFrame({
    'category': categories,
    'handoff_avg': [avg_handoff_props[cat] for cat in categories],
    'handoff_std': [std_handoff_props[cat] for cat in categories],
    'early_stop_avg': [avg_early_stop_props[cat] for cat in categories],
    'early_stop_std': [std_early_stop_props[cat] for cat in categories],
    'difference': [avg_handoff_props[cat] - avg_early_stop_props[cat] for cat in categories]
})

results_df.to_csv('average_category_statistics.csv', index=False)
print(f"\n✓ Detailed statistics saved to 'average_category_statistics.csv'")

# ==== EXPORT INDIVIDUAL PROBLEM DATA ====
individual_data = []
for i, (handoff_props, early_stop_props, info) in enumerate(zip(all_handoff_props, all_early_stop_props, problem_info)):
    for cat in categories:
        individual_data.append({
            'problem': info['name'],
            'handoff_boundary': info['handoff_boundary'],
            'early_stop_boundary': info['early_stop_boundary'],
            'total_steps': info['total_steps'],
            'category': cat,
            'handoff_proportion': handoff_props[cat],
            'early_stop_proportion': early_stop_props[cat]
        })

individual_df = pd.DataFrame(individual_data)
individual_df.to_csv('individual_problem_statistics.csv', index=False)
print(f"✓ Individual problem data saved to 'individual_problem_statistics.csv'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)