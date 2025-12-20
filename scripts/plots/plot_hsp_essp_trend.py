import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate

INPUT_PATH = Path("data/hsp_step_accuracies.jsonl")
OUTPUT_DIR = Path("data/plots/essp_hsp_plots")

NUM_PLOTS = 1

DROP_STEP = -2

CONFIG = {
    "essp_color": "#4A90E2",
    "hsp_color": "#009957",
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
    "star_color": "#E09000",
    "star_size": 300,
    "star_alpha": 1.0,
    "star_edge_color": None, #"black",
    "star_z_order": 5,
    "show_avg_star": False,
    "legend_border_pad": 1.0,
    "legend_frame_alpha": 1.0,
    # Average plot specific configs
    "num_interp_points": 50,
    "show_confidence_bands": False,
    "confidence_alpha": 0.2,
    # ESSP distribution visualization
    "show_essp_distribution": False,
    "essp_shaded_alpha": 0.1,
    "essp_shaded_y_min": 0.3,
    "essp_shaded_y_max": 0.7,
    "essp_mean_line_style": "--",
    "essp_mean_line_width": 1.5,
    "threshold_line_color": "gray",
    "threshold_line_style": "--",
    "threshold_line_width": 1.0,
    "threshold_alpha": 0.5,
}

class Plotter:
    
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

    def interpolate_to_percentage(self, essp_step_accuracies):
        """
        Interpolate essp_step_accuracies to a fixed number of percentage points.
        
        Args:
            essp_step_accuracies: List/array of accuracy values at each step
            
        Returns:
            Array of interpolated values at percentage points 0-100
        """
        n_steps = len(essp_step_accuracies)
        x_original = np.linspace(0, 100, n_steps)
        x_target = np.linspace(0, 100, self.config['num_interp_points'])
        
        f = interpolate.interp1d(x_original, essp_step_accuracies, kind='linear')
        interpolated = f(x_target)
        
        return interpolated
    
    def compute_essp_statistics(self, dataset):
        """
        Compute mean and standard deviation of ESSP positions as percentages.
        
        Returns:
            (mean_essp_pct, std_essp_pct) or (None, None) if no valid ESSPs
        """
        essp_percentages = []
        for data in dataset:
            essp_step_accuracies = data['essp_step_accuracies']
            if len(essp_step_accuracies) > 0:
                essp_step_accuracies = essp_step_accuracies[:DROP_STEP]
            if data['first_essp_index'] != -1:
                n_steps = len(essp_step_accuracies)
                essp_pct = (data['first_essp_index'] / (n_steps - 1)) * 100
                essp_percentages.append(essp_pct)
        
        if essp_percentages:
            return np.mean(essp_percentages), np.std(essp_percentages)
        return None, None
    
    def compute_hsp_statistics(self, dataset):
        """
        Compute mean and standard deviation of hsp positions as percentages.
        
        Returns:
            (mean_hsp_pct, std_hsp_pct) or (None, None) if no valid hsps
        """
        hsp_percentages = []
        for data in dataset:
            hsp_step_accuracies = data['hsp_step_accuracies']
            if len(hsp_step_accuracies) > 0:
                hsp_step_accuracies = hsp_step_accuracies[:DROP_STEP]
            if data['first_hsp_index'] != -1:
                n_steps = len(hsp_step_accuracies)
                hsp_pct = (data['first_hsp_index'] / (n_steps - 1)) * 100
                hsp_percentages.append(hsp_pct)
        
        if hsp_percentages:
            return np.mean(hsp_percentages), np.std(hsp_percentages)
        return None, None
    
    def find_threshold_crossing(self, x_values, y_values, threshold=0.5):
        """
        Find where the curve crosses a threshold value.
        
        Args:
            x_values: X-axis values (percentages)
            y_values: Y-axis values (accuracies)
            threshold: The threshold to find crossing for
            
        Returns:
            X-value where curve crosses threshold, or None if it doesn't cross
        """
        for i in range(len(y_values) - 1):
            if y_values[i] <= threshold <= y_values[i + 1]:
                x1, x2 = x_values[i], x_values[i + 1]
                y1, y2 = y_values[i], y_values[i + 1]
                x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                return x_cross
        return None
    
    def plot_essp_hsp(self, avg: bool = True):
        
        # Set matplotlib parameters
        self.set_matplotlib_parameters()
        
        # Load the dataset
        dataset = self.load_dataset()
        
        if avg:
            # Interpolate all essp_step_accuracies to the same percentage grid
            essp_interpolated_data = []
            hsp_interpolated_data = []
            for data in dataset:
                essp_step_accuracies = data['essp_step_accuracies']
                hsp_step_accuracies = data['hsp_step_accuracies']
                if len(essp_step_accuracies) > 0:
                    essp_step_accuracies = essp_step_accuracies[:DROP_STEP]
                    hsp_step_accuracies = hsp_step_accuracies[:DROP_STEP]
                if data['first_essp_index'] != -1:
                    essp_interp_acc = self.interpolate_to_percentage(essp_step_accuracies)
                    hsp_interp_acc = self.interpolate_to_percentage(hsp_step_accuracies)
                    essp_interpolated_data.append(essp_interp_acc)
                    hsp_interpolated_data.append(hsp_interp_acc)
                    
            
            # Convert to numpy array for easy computation
            essp_interpolated_data = np.array(essp_interpolated_data)
            hsp_interpolated_data = np.array(hsp_interpolated_data)
            
            # Compute mean and std error for accuracy curve
            essp_mean_accuracies = np.mean(essp_interpolated_data, axis=0)
            essp_std_error = np.std(essp_interpolated_data, axis=0) / np.sqrt(len(essp_interpolated_data))
            hsp_mean_accuracies = np.mean(hsp_interpolated_data, axis=0)
            hsp_std_error = np.std(hsp_interpolated_data, axis=0) / np.sqrt(len(hsp_interpolated_data))
            
            # Compute ESSP statistics
            mean_essp_pct, std_essp_pct = self.compute_essp_statistics(dataset)
            mean_hsp_pct, std_essp_pct = self.compute_hsp_statistics(dataset)
            
            # Create percentage x-axis
            essp_x_percentage = np.linspace(0, 100, self.config['num_interp_points'])
            hsp_x_percentage = np.linspace(0, 100, self.config['num_interp_points'])
            
            # Find where mean curve crosses 0.5 (for reference/caption)
            essp_mean_curve_crossing = self.find_threshold_crossing(essp_x_percentage, essp_mean_accuracies, threshold=0.5)
            hsp_mean_curve_crossing = self.find_threshold_crossing(hsp_x_percentage, hsp_mean_accuracies, threshold=0.5)
            
            # Plot
            plt.figure(figsize=(7, 4))
            
            # 1. Horizontal threshold line at y=0.5
            plt.axhline(
                y=0.5,
                color=self.config['threshold_line_color'],
                linestyle=self.config['threshold_line_style'],
                linewidth=self.config['threshold_line_width'],
                alpha=self.config['threshold_alpha'],
                zorder=1,
                label='Success Threshold',
            )
            
            # 2. Mean accuracy curve
            essp_line_plot = plt.plot(
                essp_x_percentage,
                essp_mean_accuracies,
                color=self.config['essp_color'],
                linewidth=self.config['line_width'],
                label='ESSP Mean Accuracy',
                zorder=3
            )[0] # Get the first (and only) line object
            hsp_line_plot = plt.plot(
                hsp_x_percentage,
                hsp_mean_accuracies,
                color=self.config['hsp_color'],
                linewidth=self.config['line_width'],
                label='HSP Mean Accuracy',
                zorder=3
            )[0] # Get the first (and only) line object
            
            # 3. Confidence bands if enabled
            if self.config['show_confidence_bands']:
                essp_band_plot = plt.fill_between(
                    essp_x_percentage,
                    essp_mean_accuracies - essp_std_error,
                    essp_mean_accuracies + essp_std_error,
                    color=self.config['essp_color'],
                    alpha=self.config['confidence_alpha'],
                    #label='±1 SE',
                    zorder=2
                )
                # Create combined label for line + band
                essp_line_plot.set_label('ESSP Mean Accuracy')
                hsp_band_plot = plt.fill_between(
                    hsp_x_percentage,
                    hsp_mean_accuracies - hsp_std_error,
                    hsp_mean_accuracies + hsp_std_error,
                    color=self.config['hsp_color'],
                    alpha=self.config['confidence_alpha'],
                    #label='±1 SE',
                    zorder=2
                )
                # Create combined label for line + band
                hsp_line_plot.set_label('HSP Mean Accuracy')
            else:
                essp_line_plot.set_label('ESSP Mean Accuracy')
                hsp_line_plot.set_label('HSP Mean Accuracy')
            
            # 4. ESSP distribution visualization
            if mean_essp_pct is not None and self.config['show_essp_distribution']:
                
                # Shaded region for ±1 SD
                plt.axvspan(
                    mean_essp_pct - std_essp_pct,
                    mean_essp_pct + std_essp_pct,
                    ymin=self.config.get('essp_shaded_y_min', 0.0),
                    ymax=self.config.get('essp_shaded_y_max', 1.0),
                    color=self.config['star_color'],
                    alpha=self.config['essp_shaded_alpha'],
                    zorder=1
                )
            
            # Vertical line at mean ESSP position
            #vert_line = plt.axvline(
            #    x=mean_essp_pct,
            #    color=self.config['star_color'],
            #    linestyle=self.config['essp_mean_line_style'],
            #    linewidth=self.config['essp_mean_line_width'],
            #    alpha=0.7,
            #    zorder=2,
            #    label='Mean ESSP Position' if not self.config['show_avg_star'] else None,
            #)
            
            # 5. Star at (mean ESSP position, 0.5)
            if mean_essp_pct is not None and self.config['show_avg_star'] is True:
                plt.scatter(
                    mean_essp_pct,
                    0.5,
                    marker='*',
                    s=self.config['star_size'],
                    color=self.config['star_color'],
                    alpha=self.config['star_alpha'],
                    edgecolors=self.config['star_edge_color'],
                    zorder=self.config['star_z_order'],
                    label='Mean ESSP Position (±1 SD)'
                )
            
            plt.xlabel('Progress Through Reasoning (%)')
            plt.ylabel('Accuracy')
            plt.title('Average Handoff & Early Stopping Success')
            
            # Get all handles and labels
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Define desired order (by label name)
            desired_order = ['HSP Mean Accuracy', 'ESSP Mean Accuracy', 'Success Threshold']
            
            # Reorder handles and labels
            ordered_handles = []
            ordered_labels = []
            for label in desired_order:
                if label in labels:
                    idx = labels.index(label)
                    ordered_handles.append(handles[idx])
                    ordered_labels.append(labels[idx])
            
            plt.legend(
                ordered_handles,
                ordered_labels,
                loc='best',
                borderpad=self.config['legend_border_pad'],
                framealpha=self.config['legend_frame_alpha'],
            )
            plt.grid(
                visible=self.config['use_grid'],
                alpha=self.config['grid_alpha'],
            )
            plt.xlim(0, 100)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            
            # Print diagnostic info
            print(f"Mean ESSP position: {mean_essp_pct:.1f}% (±{std_essp_pct:.1f}%)")
            print(f"Number of problems: {len(dataset)}")
            if essp_mean_curve_crossing is not None:
                print(f"Mean curve crosses 0.5 at: {essp_mean_curve_crossing:.1f}%")
            
            plt.savefig(f'{OUTPUT_DIR}/hsp_essp_avg_plot.pdf')
            plt.close()
        
        # Individual plots
        for i, data in enumerate(dataset):
            
            if i < NUM_PLOTS:
                
                essp_step_accuracies = data['essp_step_accuracies']
                if len(essp_step_accuracies) > 0:
                    essp_step_accuracies = essp_step_accuracies[:DROP_STEP]
                    
                first_essp_index = data['first_essp_index']
                
                plt.figure(figsize=(7, 4))
                plt.plot(
                    essp_step_accuracies,
                    color=self.config['essp_color'],
                    linewidth=self.config['line_width'],
                )
                
                if first_essp_index is not None:
                    plt.scatter(
                        first_essp_index,
                        essp_step_accuracies[first_essp_index],
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
    
    plotter = Plotter(CONFIG, INPUT_PATH)
    plotter.plot_essp_hsp(avg=True)