import json
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_PATH = Path("data/experiment_1.1_results.jsonl")
OUTPUT_DIR = Path("data/plots/essp_plots")

CONFIG = {
    "color": "#4A90E2",
    "line_width": 1.5,
    "font_family": "serif",
    "font_size": 10,
    "axes_line_width": 1.0,
    "xtick_major_width": 0.5,
    "ytick_major_width": 0.5,
    "xtick_direction": 'out',
    "ytick_direction": 'out',
    "use_grid": True,
    "grid_alpha": 0.2,
}

class FirstPointPlotter:
    
    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        
    def load_dataset(self):
        dataset = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                line_dict = json.loads(line)
                if line_dict['first_essp_index'] != -1:
                    dataset.append(json.loads(line))
        return dataset
    
    def set_matplotlib_parameters(self):
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['font.size'] = self.config['font_size']
        plt.rcParams['axes.linewidth'] = self.config['axes_line_width']
        plt.rcParams['xtick.major.width'] = self.config['xtick_major_width']
        plt.rcParams['ytick.major.width'] = self.config['ytick_major_width']
        plt.rcParams['xtick.direction'] = self.config['xtick_direction']
        plt.rcParams['ytick.direction'] = self.config['ytick_direction']
        
    def plot_scatter(self, avg: bool = False):
        """
        We want a scatter plot.
        X-axis is the first_essp_index as percentage of total steps.
        Y-axis is the first_hsp_index as percentage of total steps.
        
        What to look for:
        1. Diagonal line relationship: Does HSP = ESSP?
        2. HSP > ESSP region: Points above the diagonal mean the model "knows" the answer (ESSP) before it can reliably hand it off.
        3. HSP < ESSP region: Points beow the diagonal line mean the model can handoff the answer successfully before it "knows" it.
        """
        
        # Set matplotlib parameters
        self.set_matplotlib_parameters()
        
        # Load the dataset
        dataset = self.load_dataset()
        
        first_essp_indices = [data['first_essp_index'] for data in dataset]
        total_steps = [data['toral_steps'] for data in dataset]
        
        
if __name__ == "__main__":
    
    plotter = FirstPointPlotter(CONFIG, INPUT_PATH)
    print(plotter.plot_scatter(avg=False))
    
