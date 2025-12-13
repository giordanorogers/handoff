import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('experiment_1_results.jsonl', lines=True)

df['essp_proportion'] = df['essp_index'].replace(-1, pd.NA) / df['total_steps']

df_clean = df.dropna(subset=['essp_proportion'])

# Calculate mean for average lines
mean_essp = df_clean['essp_proportion'].mean()

# Histogram showing the distribution of ESSP proportions
plt.figure(figsize=(10, 6))
plt.hist(df_clean['essp_proportion'], bins=25, edgecolor='black')
plt.axvline(mean_essp, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_essp:.2f}')
plt.xlabel('ESSP Proportion')
plt.ylabel('Frequency')
plt.title('Distribution of ESSP Proportions')
plt.legend()
plt.tight_layout()
plt.savefig("histogram.png")

plt.figure(figsize=(10, 6))



# Sort the ESSP proportions
sorted_proportions = np.sort(df_clean['essp_proportion'])

# Calculate cumulative proportion (y-axis)
cumulative_prop = np.arange(1, len(sorted_proportions) + 1) / len(sorted_proportions)

# Plot
plt.plot(sorted_proportions, cumulative_prop, linewidth=2)
plt.axvline(mean_essp, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_essp:.2f}')
plt.xlabel('Percentage Through Prompt')
plt.ylabel('Proportion of Problems Reached ESSP')
plt.title('Cumulative ESSP Arrival')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("cumulative_essp.png")

plt.show()