import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('./scripts/plots/handoff_problem2.csv')

# Define categories and colors
categories = ['planning', 'exploration', 'reasoning', 'insight', 'monitoring', 'correction']
colors = {
    'planning': '#3498db',      # blue
    'exploration': '#2ecc71',   # green
    'reasoning': '#e74c3c',     # red
    'insight': '#f39c12',       # orange/gold
    'monitoring': '#9b59b6',    # purple
    'correction': '#95a5a6'     # gray
}

# Split data: handoff-sufficient vs ALL early-stop sufficient
handoff = df[df['is_handoff_sufficient'] == True]
early_stop_all = df[df['is_early_stop_sufficient'] == True]

# Calculate proportions
def get_proportions(subset):
    counts = subset['behavioral_category'].value_counts()
    total = len(subset)
    return {cat: counts.get(cat, 0) / total for cat in categories}

handoff_props = get_proportions(handoff)
early_stop_props = get_proportions(early_stop_all)

# ==== FIGURE 1: Grouped Bar Chart with Proportions ====
fig1, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, [handoff_props[cat] for cat in categories], 
               width, label='Handoff-sufficient (steps 1-72)',
               color=[colors[cat] for cat in categories], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, [early_stop_props[cat] for cat in categories], 
               width, label='Early-stop sufficient (steps 1-220)',
               color=[colors[cat] for cat in categories], alpha=0.4, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Proportion of Steps', fontsize=12)
ax.set_xlabel('Behavioral Category', fontsize=12)
ax.set_title('Problem 2: Distribution of Behavioral Categories by Region', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
ax.legend(frameon=True, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, max(max(handoff_props.values()), max(early_stop_props.values())) * 1.15)

# Add value labels on bars (as percentages)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.02:  # Only label if >2%
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('problem2_category_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==== FIGURE 2: Timeline View ====
fig2, ax = plt.subplots(figsize=(14, 5))

step_nums = df['step_number'].values
category_nums = [categories.index(cat) for cat in df['behavioral_category']]
category_colors = [colors[cat] for cat in df['behavioral_category']]

# Plot as horizontal bands
for i, (step, cat_idx, color) in enumerate(zip(step_nums, category_nums, category_colors)):
    ax.barh(cat_idx, 1, left=step-0.5, height=0.8, color=color, edgecolor='white', linewidth=0.3)

# Add vertical line at handoff boundary
handoff_boundary = handoff['step_number'].max()
ax.axvline(x=handoff_boundary + 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.text(handoff_boundary + 0.5, len(categories)-0.3, f'Handoff boundary (step {handoff_boundary})', 
        ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Step Number', fontsize=12)
ax.set_ylabel('Behavioral Category', fontsize=12)
ax.set_title('Problem 2: Sequential Flow of Reasoning Categories', fontsize=14, pad=15)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels([cat.capitalize() for cat in categories])
ax.set_xlim(0, len(df) + 1)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('problem2_category_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

# ==== Print Summary Statistics ====
print("\n=== PROBLEM 2 SUMMARY STATISTICS ===\n")
print(f"Handoff-sufficient region (steps 1-{handoff_boundary}):")
for cat in categories:
    count = handoff['behavioral_category'].value_counts().get(cat, 0)
    pct = handoff_props[cat] * 100
    print(f"  {cat.capitalize()}: {count} steps ({pct:.1f}%)")

print(f"\nTotal handoff steps: {len(handoff)}")

early_stop_boundary = early_stop_all['step_number'].max()
print(f"\nEarly-stop sufficient region (steps 1-{early_stop_boundary}):")
for cat in categories:
    count = early_stop_all['behavioral_category'].value_counts().get(cat, 0)
    pct = early_stop_props[cat] * 100
    print(f"  {cat.capitalize()}: {count} steps ({pct:.1f}%)")

print(f"\nTotal early-stop steps: {len(early_stop_all)}")

print("\n=== KEY OBSERVATIONS ===")
insight_steps = df[df['behavioral_category'] == 'insight']['step_number'].values
print(f"Insight steps: {list(insight_steps)}")
print(f"Insights in handoff region: {sum(s <= handoff_boundary for s in insight_steps)}/{len(insight_steps)}")

correction_steps = df[df['behavioral_category'] == 'correction']['step_number'].values
print(f"Correction steps: {list(correction_steps)}")
print(f"Corrections in handoff region: {sum(s <= handoff_boundary for s in correction_steps)}/{len(correction_steps)}")

print("\n=== PROPORTIONAL DIFFERENCES ===")
print("Categories enriched in handoff vs full early-stop:")
for cat in categories:
    diff = (handoff_props[cat] - early_stop_props[cat]) * 100
    if abs(diff) > 2:  # Only show differences > 2%
        direction = "higher" if diff > 0 else "lower"
        print(f"  {cat.capitalize()}: {abs(diff):.1f}% {direction} in handoff region")

# ==== BONUS: Comparison with Problem 1 ====
# If you have problem 1 data loaded as df_p1, you can compare:
print("\n=== COMPARISON: PROBLEM 1 vs PROBLEM 2 ===")
print("\nProblem 2 characteristics:")
print(f"  - Much longer trace: {len(df)} steps (vs 73 in Problem 1)")
print(f"  - Reasoning-dominated: {early_stop_props['reasoning']*100:.1f}% reasoning")
print(f"  - Fewer insights: {len(insight_steps)} total")
print(f"  - All insights AFTER handoff boundary (steps {list(insight_steps)})")
print(f"\nHypothesis: Problem 2 requires different handoff mechanism")
print(f"  - P1: Handoff works by transferring key insights (steps 32-33)")
print(f"  - P2: Handoff works by transferring graph structure, NOT insights")