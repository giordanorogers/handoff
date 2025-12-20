import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data/hsp_step_accuracies.jsonl")
OUTPUT_DIR = Path("data/plots/essp_hsp_plots")

CONFIG = {
    "essp_color": "#690A8F",
    "hsp_color": "#009957",
    "scatter_alpha": 0.6,
    "scatter_size": 50,
    "font_family": "serif",
    "font_size": 10,
    "axes_line_width": 1.0,
    "xtick_major_width": 0.5,
    "ytick_major_width": 0.5,
    "xtick_direction": 'out',
    "ytick_direction": 'out',
    "use_grid": True,
    "grid_alpha": 0.2,
    "diagonal_color": "gray",
    "diagonal_style": "--",
    "diagonal_width": 1.5,
    "diagonal_alpha": 0.7,
    "annotation_font_size": 12,
    "annotation_color": "gray",
}

# Set matplotlib parameters
plt.rcParams['font.family'] = CONFIG['font_family']
plt.rcParams['font.size'] = CONFIG['font_size']
plt.rcParams['axes.linewidth'] = CONFIG['axes_line_width']
plt.rcParams['xtick.major.width'] = CONFIG['xtick_major_width']
plt.rcParams['ytick.major.width'] = CONFIG['ytick_major_width']
plt.rcParams['xtick.direction'] = CONFIG['xtick_direction']
plt.rcParams['ytick.direction'] = CONFIG['ytick_direction']

# Load data
essp_positions = []
hsp_positions = []

with open(INPUT_PATH, 'r') as f:
    for line in f:
        data = json.loads(line)
        
        # Skip if no valid ESSP
        if data['first_essp_index'] == -1:
            continue
            
        # Get step counts
        essp_steps = data['essp_step_accuracies'][:-2]  # DROP_STEP = -2
        hsp_steps = data['hsp_step_accuracies'][:-2]
        
        # Calculate proportions (0 to 100%)
        essp_pct = (data['first_essp_index'] / (len(essp_steps) - 1)) * 100
        hsp_pct = (data['first_hsp_index'] / (len(hsp_steps) - 1)) * 100
        
        essp_positions.append(essp_pct)
        hsp_positions.append(hsp_pct)

# Create scatter plot
plt.figure(figsize=(7, 6))

# Plot diagonal line
plt.plot(
    [0, 100], 
    [0, 100], 
    linestyle=CONFIG['diagonal_style'],
    linewidth=CONFIG['diagonal_width'],
    color=CONFIG['diagonal_color'],
    alpha=CONFIG['diagonal_alpha'],
    zorder=1
)

# Plot scatter points
plt.scatter(
    essp_positions, 
    hsp_positions, 
    alpha=CONFIG['scatter_alpha'], 
    s=CONFIG['scatter_size'],
    color=CONFIG['essp_color'],
    zorder=2
)

# Add annotations
plt.text(
    25, 75,
    'HSP later than ESSP',
    fontsize=CONFIG['annotation_font_size'],
    color=CONFIG['annotation_color'],
    style='italic',
    ha='center',
    va='center',
    rotation=0
)

plt.text(
    75, 25,
    'ESSP later than HSP',
    fontsize=CONFIG['annotation_font_size'],
    color=CONFIG['annotation_color'],
    style='italic',
    ha='center',
    va='center',
    rotation=0
)

plt.xlabel('ESSP Position (% through reasoning)')
plt.ylabel('HSP Position (% through reasoning)')
plt.title('ESSP vs HSP Positions')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(
    visible=CONFIG['use_grid'],
    alpha=CONFIG['grid_alpha']
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/essp_vs_hsp_scatter.pdf')
plt.close()

print(f"Plotted {len(essp_positions)} problems")