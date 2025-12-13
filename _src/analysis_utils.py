import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def normalize_trace_length(hsr_curve: list[dict], num_points: int = 100) -> np.ndarray:
    """
    Interpolates an HSR curve to a fixed number of points (0% to 100% progress).
    This allows averaging curves across problems with different step counts.
    """
    if not hsr_curve:
        return np.zeros(num_points)

    # Extract x (progress 0..1) and y (HSR score)
    steps = np.array([item['step_idx'] for item in hsr_curve])
    scores = np.array([item['hsr'] for item in hsr_curve])
    
    if len(steps) < 2:
        # If trace is too short, return constant value
        return np.full(num_points, scores[0] if len(scores) > 0 else 0.0)

    # Normalize steps to 0..1
    max_step = steps[-1]
    if max_step == 0:
        return np.full(num_points, scores[0])
        
    normalized_steps = steps / max_step
    
    # Interpolate to fixed grid
    target_grid = np.linspace(0, 1, num_points)
    f = interp1d(normalized_steps, scores, kind='linear', bounds_error=False, fill_value=(scores[0], scores[-1]))
    
    return f(target_grid)

def aggregate_results(results_data: list[dict]) -> pd.DataFrame:
    """
    Processes raw experiment results into a DataFrame for plotting.
    """
    coherent_curves = []
    incoherent_curves = []
    
    for item in results_data:
        # Normalize both curves to 100 points
        c_curve = normalize_trace_length(item['hsr_coherent'])
        i_curve = normalize_trace_length(item['hsr_incoherent'])
        
        coherent_curves.append(c_curve)
        incoherent_curves.append(i_curve)
        
    # Convert to arrays for easy averaging
    coherent_arr = np.array(coherent_curves)   # Shape: (N_problems, 100)
    incoherent_arr = np.array(incoherent_curves) # Shape: (N_problems, 100)
    
    # Create a DataFrame for plotting
    x_axis = np.linspace(0, 100, 100) # Percentage progress
    
    df_coherent = pd.DataFrame({
        'progress': x_axis,
        'hsr_mean': np.mean(coherent_arr, axis=0),
        'hsr_std': np.std(coherent_arr, axis=0),
        'condition': 'Coherent'
    })
    
    df_incoherent = pd.DataFrame({
        'progress': x_axis,
        'hsr_mean': np.mean(incoherent_arr, axis=0),
        'hsr_std': np.std(incoherent_arr, axis=0),
        'condition': 'Shuffled (Incoherent)'
    })
    
    return pd.concat([df_coherent, df_incoherent], ignore_index=True)
