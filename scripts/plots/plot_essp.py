import json
import numpy as np
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt

INPUT_PATH = Path("data/experiment_1.1_results.jsonl")
OUTPUT_DIR = Path("data/plots/essp_plots")

NUM_PLOTS = 1

CONFIG = {
    "line_color": "#4A90E2",
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
    "star_color": "#F5A623",
    "star_size": 300,
    "star_alpha": 1.0,
    "star_edge_color": "black",
    "star_z_order": 5,
    "legend_border_pad": 1.0,
    "legend_frame_alpha": 1.0,
    "num_interp_points": 100,
    "show_confidence_bands": False,
    "confidence_alpha": 0.2,
}

class ESSPPlotter:
    
    def __init__(self, config, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        
    def load_dataset(self):
        dataset = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
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

    def interpolate_to_percentage(self, step_accuracies):
        """
        Interpolate step_accuracies to a fixed number of percentage points.
        Args:
            step_accuracies: List/array of accuracy values at each step.
        Returns:
            Array of interpolated values at percentage points 0-100%
        """
        n_steps = len(step_accuracies)
        # Original x values as percentages (0 to 100)
        x_original = np.linspace(0, 100, n_steps)
        # Target x values (uniform percentage grid)
        x_target = np.linspace(0, 100, self.config['num_interp_points'])
        
        # Linear interpolation
        f = interpolate.interp1d(x_original, step_accuracies, kind='linear')
        interpolated = f(x_target)
        
        return interpolated
    
    def compute_average_essp_percentage(self, dataset):
        """
        Compute the average first_essp_index as a percentage of total steps.
        """
        essp_percentages = []
        for data in dataset:
            if data['first_essp_index'] != -1:
                n_steps = len(data['step_accuracies'])
                essp_pct = (data['first_essp_index'] / (n_steps - 1)) * 100
                essp_percentages.append(essp_pct)
                
        return np.mean(essp_percentages)
    
    def plot_essp(self, avg: bool = True):
        
        # Set matplotlib parameters
        self.set_matplotlib_parameters()
        
        # Load the dataset
        dataset = self.load_dataset()
        
        if avg:
            
            # Interpolate all step_accuracies to the same percentage grid
            interpolated_data = []
            for data in dataset:
                interp_acc = self.interpolate_to_percentage(data['step_accuracies'])
                interpolated_data.append(interp_acc)
                
            # Convert to numpy array for easy consumption
            interpolated_data = np.array(interpolated_data) # Shape: (n_problems, n_points)
            
            # Compute mean and std error
            mean_accuracies = np.mean(interpolated_data, axis=0)
            std_error = np.std(interpolated_data, axis=0) / np.sqrt(len(interpolated_data))
            
            # Compute average ESSP position
            avg_essp_pct = self.compute_average_essp_percentage(dataset)
            
            # Create percentage x-axis
            x_percentage = np.linspace(0, 100, self.config['num_interp_points'])
            
            # Plot
            plt.figure(figsize=(7,4))
            plt.plot(
                x_percentage,
                mean_accuracies,
                color=self.config['line_color'],
                linewidth=self.config['line_width'],
                label='Mean Accuracy'
            )
            
            # Add confidence bands if enabled
            if self.config['show_confidence_bands']:
                plt.fill_between(
                    x_percentage,
                    mean_accuracies - std_error,
                    mean_accuracies + std_error,
                    color=self.config['line_color'],
                    alpha=self.config['confidence_alpha'],
                    label='±1 SE'
                )
                
            # Add star at average ESSP position
            if avg_essp_pct is not None:
                # Find the closes percentage point to avg_essp_pct
                closest_idx = np.argmin(np.abs(x_percentage - avg_essp_pct))
                plt.scatter(
                    x_percentage[closest_idx],
                    mean_accuracies[closest_idx],
                    marker='*',
                    s=self.config['star_size'],
                    color=self.config['star_color'],
                    alpha=self.config['star_alpha'],
                    edgecolors=self.config['star_edge_color'],
                    zorder=self.config['star_z_order'],
                    label='Average First ESSP'
                )
                
            plt.xlabel('Progress Through Reasoning (%)')
            plt.ylabel('Accuracy')
            plt.title('Average Early Stopping Success')
            plt.legend(
                loc='best',
                borderpad=self.config['legend_border_pad'],
                framealpha=self.config['legend_frame_alpha'],
            )
            plt.grid(
                visible=self.config['use_grid'],
                alpha=self.config['grid_alpha'],
            )
            plt.xlim(0, 100)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/essp_avg_plot.pdf')
            plt.close()
        
        for i, data in enumerate(dataset):
            
            # Make a simple plot of the first element
            if i < NUM_PLOTS:
                
                step_accuracies = data['step_accuracies']
                first_essp_index = data['first_essp_index']
                
                plt.figure(figsize=(7,4))
                plt.plot(
                    step_accuracies,
                    color=self.config['line_color'],
                    linewidth=self.config['line_width'],
                )
                plt.scatter(
                    first_essp_index,
                    step_accuracies[first_essp_index],
                    marker='*',
                    s=self.config['star_size'],
                    color=self.config['star_color'],
                    alpha=self.config['star_alpha'],
                    edgecolors=self.config['star_edge_color'],
                    zorder=self.config['star_z_order'],
                    label="First ESSP"
                )
                plt.xlabel('Number of Segments')
                plt.ylabel('Accuracy')
                plt.title(f'Early Stopping Success: Problem {i}')
                plt.legend(
                    loc='best',
                    borderpad=self.config['legend_border_pad'],
                    framealpha=self.config['legend_frame_alpha'],
                )
                plt.grid(
                    visible=self.config['use_grid'],
                    alpha=self.config['grid_alpha'],
                )
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/essp_plot{i}.pdf')
                plt.close()
    
if __name__ == "__main__":
    
    plotter = ESSPPlotter(CONFIG, INPUT_PATH)
    plotter.plot_essp(avg=True)
