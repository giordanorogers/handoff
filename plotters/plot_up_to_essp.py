import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

df = pd.read_json('experiment_1_results.jsonl', lines=True)

# Filter to problems with ESSP
df_clean = df[df['essp_index'] != -1].copy()

# Average correctness up to ESSP
plt.figure(figsize=(10, 6))

normalized_x = np.linspace(0, 1, 100)
interpolated_y = []

for _, row in df_clean.iterrows():
    accs = row['step_accuracies']
    if len(accs) < 2:
        continue
    
    # Normalize from 0 to 1 representing 0-100% of journey to ESSP
    x_original = np.linspace(0, 1, len(accs))
    
    # Interpolate onto standard grid
    f = interp1d(x_original, accs, kind='linear', bounds_error=False, 
                 fill_value=(accs[0], accs[-1]))
    y_interp = f(normalized_x)
    interpolated_y.append(y_interp)

# Calculate mean and std
y_matrix = np.array(interpolated_y)
y_mean = np.mean(y_matrix, axis=0)
y_std = np.std(y_matrix, axis=0)

# Plot
plt.plot(normalized_x * 100, y_mean, color='blue', linewidth=2, label='Mean Accuracy')
plt.fill_between(normalized_x * 100, 
                 np.maximum(0, y_mean - y_std), 
                 np.minimum(1, y_mean + y_std), 
                 color='blue', alpha=0.2, label='Std Dev')
plt.axhline(y=0.5, color='r', linestyle='--', label='Success Threshold')
plt.xlabel('Progress to ESSP (%)')
plt.ylabel('Average P(Correct)')
plt.title(f'Average Correctness Until ESSP (N={len(interpolated_y)})')
plt.legend()
#plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('progress_to_essp.png')
plt.show()