"""Utility functions for UMAP visualization and analysis.

This module provides functions for visualizing and analyzing the topological
structure of channel data using UMAP dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Optional

def plot_umap_embeddings(embeddings: np.ndarray, 
                        labels: np.ndarray,
                        model_names: list,
                        full_model_list: list = None,
                        plot_points: Optional[int] = None,
                        add_labels: bool = True,
                        title: str = None,
                        xlim: tuple = None,
                        ylim: tuple = None) -> None:
    """
    Plot UMAP embeddings with model labels and color coding.
    
    Args:
        embeddings: UMAP embeddings
        labels: Array of labels for each point
        model_names: List of model names to plot
        full_model_list: Complete list of all possible models (for consistent color mapping)
        plot_points: Number of points to plot per model
        add_labels: Whether to add model name labels
        title: Title for the plot
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
    """
    # If full_model_list is not provided, use model_names as the complete list
    if full_model_list is None:
        full_model_list = model_names
    
    # Create a color mapping based on the full model list
    n_full_models = len(full_model_list)
    model_type_color_map = {}
    
    # Assign colors based on position in the full model list
    for i, model in enumerate(full_model_list):
        # Use a value between 0 and 1 for the viridis colormap
        model_type_color_map[model] = i / max(1, n_full_models - 1)
    
    # Create a color mapping for the current models
    n_models = len(model_names)
    colors = []
    for model in model_names:
        if model in model_type_color_map:
            colors.append(plt.cm.viridis(model_type_color_map[model]))
        else:
            # For any model not in our predefined map, assign a color based on position
            colors.append(plt.cm.viridis(len(colors) / max(n_models, 1)))
    
    custom_cmap = plt.cm.colors.ListedColormap(colors)
    
    # Create a mapping from model names to their indices in the full list
    model_to_full_index = {model: i for i, model in enumerate(full_model_list)}
    
    # Select random points for plotting
    if not plot_points:
        plot_indices = np.arange(len(labels))
    else:
        plot_indices = []
        for i, model in enumerate(model_names):
            # Get the index of this model in the full list
            full_idx = model_to_full_index.get(model)
            if full_idx is None:
                continue
                
            # Look for labels that match the index in the full list
            class_mask = labels == full_idx
            n_points = np.sum(class_mask)
            
            # Skip if there are no points for this class
            if n_points == 0:
                continue
                
            all_data_idxs = np.where(class_mask)[0]
            random_data_idxs = np.random.choice(all_data_idxs, size=min(n_points, plot_points), replace=False)
            plot_indices.extend(random_data_idxs)
        
        # Check if we have any points to plot
        if not plot_indices:
            print("Warning: No points to plot for any of the classes.")
            return
            
        plot_indices = np.array(plot_indices)
    
    plt.figure(figsize=(7, 5), dpi=200, tight_layout=True)
    
    # Map the actual labels to indices for coloring
    plot_labels = np.array([model_names.index(full_model_list[int(label)]) for label in labels[plot_indices]])
    
    scatter = plt.scatter(embeddings[plot_indices, 0], embeddings[plot_indices, 1],
                         c=plot_labels, cmap=custom_cmap, s=10, alpha=0.7)

    # Add model name labels
    if add_labels:
        for i, model in enumerate(model_names):
            class_mask = labels == model_to_full_index.get(model, i)
            if np.any(class_mask):
                mean_x = np.mean(embeddings[class_mask, 0])
                mean_y = np.mean(embeddings[class_mask, 1])
                
                plt.annotate(model, xy=(mean_x, mean_y),
                            xytext=(mean_x + 1, mean_y + 1),
                            color=colors[i], fontsize=9, weight='bold',
                            bbox=dict(facecolor='grey', alpha=0.7, edgecolor='none', pad=0.5),
                            arrowprops=dict(facecolor=colors[i], shrink=0.05, width=1, headwidth=5))

    # Improve colorbar formatting
    tick_locs = np.arange(n_models)
    scatter.set_clim(-0.5, n_models - 0.5)

    # Create colorbar with centered ticks
    cbar = plt.colorbar(scatter, label='Channel Model', ticks=tick_locs,
                       boundaries=np.arange(-0.5, n_models + 0.5), values=np.arange(n_models))
    cbar.set_ticklabels(model_names)

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid()
    # plt.show()  # Remove this line to let caller control when to show 

def plot_umap_3d_histogram(embeddings: np.ndarray,
                          labels: np.ndarray,
                          model_names: list,
                          full_model_list: list = None,
                          n_bins: int = 50,
                          plot_points: Optional[int] = None,
                          title: str = None,
                          xlim: tuple = None,
                          ylim: tuple = None,
                          alpha: float = 0.7,
                          normalize: bool = True,
                          density_threshold: float = 1.0,
                          view_angle: tuple = (30, 45)) -> None:
    """
    Create a 3D histogram/mesh visualization of UMAP embeddings.
    
    Args:
        embeddings: UMAP embeddings of shape (n_samples, 2)
        labels: Array of labels for each point
        model_names: List of model names to plot
        full_model_list: Complete list of all possible models (for consistent color mapping)
        n_bins: Number of bins for the histogram in each dimension
        plot_points: Number of points to plot per model
        title: Title for the plot
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
        alpha: Transparency of the mesh surface
        normalize: Whether to normalize each model's density to its maximum (default: True)
        density_threshold: Minimum density value to display (in %, default: 1.0)
        view_angle: Tuple of (elevation, azimuth) angles for the 3D view
    """
    # If full_model_list is not provided, use model_names as the complete list
    if full_model_list is None:
        full_model_list = model_names
    
    # Create a color mapping based on the full model list
    n_full_models = len(full_model_list)
    model_type_color_map = {}
    
    # Assign colors based on position in the full model list
    for i, model in enumerate(full_model_list):
        model_type_color_map[model] = i / max(1, n_full_models - 1)
    
    # Create a color mapping for the current models
    n_models = len(model_names)
    colors = []
    for model in model_names:
        if model in model_type_color_map:
            colors.append(plt.cm.viridis(model_type_color_map[model]))
        else:
            colors.append(plt.cm.viridis(len(colors) / max(n_models, 1)))
    
    # Create a mapping from model names to their indices in the full list
    model_to_full_index = {model: i for i, model in enumerate(full_model_list)}
    
    # Select points for plotting
    if not plot_points:
        plot_indices = np.arange(len(labels))
    else:
        plot_indices = []
        for i, model in enumerate(model_names):
            full_idx = model_to_full_index.get(model)
            if full_idx is None:
                continue
            
            class_mask = labels == full_idx
            n_points = np.sum(class_mask)
            
            if n_points == 0:
                continue
                
            all_data_idxs = np.where(class_mask)[0]
            random_data_idxs = np.random.choice(all_data_idxs, size=min(n_points, plot_points), replace=False)
            plot_indices.extend(random_data_idxs)
        
        if not plot_indices:
            print("Warning: No points to plot for any of the classes.")
            return
            
        plot_indices = np.array(plot_indices)
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set view angle
    ax.view_init(*view_angle)
    
    # For each model, create a separate histogram
    for i, model in enumerate(model_names):
        full_idx = model_to_full_index.get(model)
        if full_idx is None:
            continue
            
        # Get points for this model
        model_mask = labels[plot_indices] == full_idx
        if not np.any(model_mask):
            continue
            
        model_points = embeddings[plot_indices][model_mask]
        
        # Create histogram
        if xlim is None:
            x_range = (model_points[:, 0].min(), model_points[:, 0].max())
        else:
            x_range = xlim
            
        if ylim is None:
            y_range = (model_points[:, 1].min(), model_points[:, 1].max())
        else:
            y_range = ylim
            
        hist, xedges, yedges = np.histogram2d(
            model_points[:, 0], model_points[:, 1],
            bins=n_bins, range=[x_range, y_range]
        )
        
        # Normalize histogram if requested
        if normalize:
            hist = hist / hist.max() * 100  # Convert to percentage
        
        # Create meshgrid for surface plot
        xpos, ypos = np.meshgrid(xedges[:-1] + np.diff(xedges)/2,
                                yedges[:-1] + np.diff(yedges)/2)
        
        # Create mask for low density regions
        mask = hist < density_threshold
        
        # Create a masked array for the surface
        hist_masked = np.ma.masked_array(hist, mask=mask)
        
        # Create surface plot
        ax.plot_surface(xpos, ypos, hist_masked.T, color=colors[i], alpha=alpha, shade=True,
                        lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
    
    # Set labels and title
    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_zlabel('Density (%)' if normalize else 'Density')
    
    if title:
        plt.title(title)
        
    # Add a custom legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], alpha=alpha)
                      for i in range(len(model_names))]
    ax.legend(legend_elements, model_names, 
             loc='upper center', 
             bbox_to_anchor=(0.5, 0.85),
             ncol=len(model_names),
             frameon=True,
             fancybox=True,
             shadow=True)
    
    plt.tight_layout() 

def plot_amplitude_distribution(amplitudes: np.ndarray, labels: np.ndarray, model_names: list,
                                full_model_list: list = None, n_bins: int = 50, title: str = None,
                                density: bool = True) -> None:
    """
    Plot the amplitude distribution histogram of the data.
    
    Args:
        data_real: Real data matrix of shape (n_samples, n_features)
        labels: Array of labels for each point
        model_names: List of model names to plot
        full_model_list: Complete list of all possible models (for consistent color mapping)
        n_bins: Number of bins for the histogram
        title: Title for the plot
        density: Whether to normalize the histogram to form a probability density
    """
    # If full_model_list is not provided, use model_names as the complete list
    if full_model_list is None:
        full_model_list = model_names
    
    # Create a color mapping based on the full model list
    n_full_models = len(full_model_list)
    model_type_color_map = {}
    
    # Assign colors based on position in the full model list
    for i, model in enumerate(full_model_list):
        model_type_color_map[model] = i / max(1, n_full_models - 1)
    
    # Create a color mapping for the current models
    n_models = len(model_names)
    colors = []
    for model in model_names:
        if model in model_type_color_map:
            colors.append(plt.cm.viridis(model_type_color_map[model]))
        else:
            colors.append(plt.cm.viridis(len(colors) / max(n_models, 1)))
    
    # Create figure
    plt.figure(figsize=(10, 6), dpi=200)
    
    # Plot histogram for each model
    for i, model in enumerate(model_names):
        model_idx = full_model_list.index(model)
        model_mask = labels == model_idx
        model_amplitudes = amplitudes[model_mask]
        
        plt.hist(model_amplitudes, bins=n_bins, alpha=0.5, 
                 label=model, color=colors[i], density=density)
    
    # Set labels and title
    plt.xlabel('Amplitude')
    plt.ylabel('Density' if density else 'Count')
    if title:
        plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show()  # Remove this line to let caller control when to show

