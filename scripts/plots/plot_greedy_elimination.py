import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths - adjust as needed
PROBLEM_0_PATH = Path("data/greedy_elimination_results/problem_0_final.json")
PROBLEM_1_PATH = Path("data/greedy_elimination_results/problem_1_final.json")
OUTPUT_DIR = Path("data/plots/elimination_plots")

CONFIG = {
    # Line colors
    "problem_0_color": "#4A90E2",  # Blue (matches your ESSP color)
    "problem_1_color": "#009957",  # Green (matches your HSP color)
    "threshold_line_color": "gray",
    
    # Line styles
    "line_width": 1.5,
    "threshold_line_style": "--",
    "threshold_line_width": 1.0,
    "threshold_alpha": 0.5,
    
    # Markers
    "marker_size": 4,
    "marker_style": "o",
    
    # Font settings
    "font_family": "serif",
    "font_size": 10,
    
    # Axes
    "axes_line_width": 1.0,
    "xtick_major_width": 0.5,
    "ytick_major_width": 0.5,
    "xtick_direction": "out",
    "ytick_direction": "out",
    
    # Grid
    "use_grid": True,
    "grid_alpha": 0.2,
    
    # Legend
    "legend_border_pad": 1.0,
    "legend_frame_alpha": 1.0,
    
    # Figure
    "figsize": (7, 4),
}


class EliminationPlotter:
    
    def __init__(self, config):
        self.config = config
        
    def set_matplotlib_parameters(self):
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['font.size'] = self.config['font_size']
        plt.rcParams['axes.linewidth'] = self.config['axes_line_width']
        plt.rcParams['xtick.major.width'] = self.config['xtick_major_width']
        plt.rcParams['ytick.major.width'] = self.config['ytick_major_width']
        plt.rcParams['xtick.direction'] = self.config['xtick_direction']
        plt.rcParams['ytick.direction'] = self.config['ytick_direction']
    
    def load_elimination_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def extract_curve_data(self, data):
        """
        Extract (fraction_retained, accuracy) pairs from elimination history.
        
        Returns:
            fractions: list of fraction retained values
            accuracies: list of corresponding accuracies
        """
        initial_count = len(data['initial_indices'])
        baseline_accuracy = data['baseline_accuracy']
        
        fractions = [1.0]
        accuracies = [baseline_accuracy]
        
        for entry in data['elimination_history']:
            if entry.get('removed') is not None:
                remaining = entry['remaining_count']
                accuracy = entry['accuracy_after']
                fraction = remaining / initial_count
                fractions.append(fraction)
                accuracies.append(accuracy)
        
        # Add final state
        final_count = len(data['final_indices'])
        final_accuracy = data['final_accuracy']
        final_fraction = final_count / initial_count
        
        # Only add if different from last entry
        if fractions[-1] != final_fraction:
            fractions.append(final_fraction)
            accuracies.append(final_accuracy)
        
        return np.array(fractions), np.array(accuracies)
    
    def plot_elimination_curves(self, problem_paths, output_path):
        """
        Plot elimination curves for multiple problems.
        
        Args:
            problem_paths: list of (path, label, color) tuples
            output_path: where to save the figure
        """
        self.set_matplotlib_parameters()
        
        plt.figure(figsize=self.config['figsize'])
        
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
        
        # 2. Plot each problem's curve
        for path, label, color in problem_paths:
            data = self.load_elimination_data(path)
            fractions, accuracies = self.extract_curve_data(data)
            
            # Convert to percentage removed for x-axis
            percentages_removed = (1 - fractions) * 100
            
            plt.plot(
                percentages_removed,
                accuracies,
                color=color,
                linewidth=self.config['line_width'],
                marker=self.config['marker_style'],
                markersize=self.config['marker_size'],
                label=label,
                zorder=3,
            )
            
            # Print summary stats
            initial_count = len(data['initial_indices'])
            final_count = len(data['final_indices'])
            peak_acc = max(accuracies)
            peak_idx = np.argmax(accuracies)
            peak_retention = fractions[peak_idx]
            
            print(f"{label}:")
            print(f"  Compression: {initial_count} → {final_count} sentences ({final_count/initial_count*100:.1f}%)")
            print(f"  Peak accuracy: {peak_acc:.0%} at {peak_retention:.0%} retention")
            print(f"  Baseline accuracy: {data['baseline_accuracy']:.0%}")
            print(f"  Final accuracy: {data['final_accuracy']:.0%}")
            print()
        
        # 3. Configure axes
        plt.xlabel('Fraction of Sentences Removed (%)')
        plt.ylabel('Handoff Accuracy')
        plt.title('Greedy Backward Elimination: Compression vs. Transfer Accuracy')
        
        plt.xlim(0, 100)
        plt.ylim(0, 1.0)
        
        # 4. Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Reorder: problems first, then threshold
        desired_order = [l for l in labels if 'Problem' in l] + ['Success Threshold']
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
        
        # 5. Grid
        plt.grid(
            visible=self.config['use_grid'],
            alpha=self.config['grid_alpha'],
        )
        
        plt.tight_layout()
        
        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.savefig(output_path.with_suffix('.pdf'))
        print(f"Saved to {output_path}")
        plt.close()
    
    def plot_single_problem(self, problem_path, label, color, output_path):
        """Plot a single problem's elimination curve."""
        self.plot_elimination_curves(
            [(problem_path, label, color)],
            output_path
        )


if __name__ == "__main__":
    plotter = EliminationPlotter(CONFIG)
    
    # Combined plot
    problem_paths = [
        (PROBLEM_0_PATH, "Problem 1 (39→4 sentences)", CONFIG['problem_0_color']),
        (PROBLEM_1_PATH, "Problem 2 (16→4 sentences)", CONFIG['problem_1_color']),
    ]
    
    plotter.plot_elimination_curves(
        problem_paths,
        OUTPUT_DIR / "elimination_curves_combined.png"
    )
    
    # Individual plots (optional)
    # plotter.plot_single_problem(
    #     PROBLEM_0_PATH,
    #     "Problem 0",
    #     CONFIG['problem_0_color'],
    #     OUTPUT_DIR / "elimination_curve_p0.png"
    # )