"""Utility functions for UMAP visualization and analysis.

This module provides functions for visualizing and analyzing the topological
structure of channel data using UMAP dimensionality reduction.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_umap_embeddings(embeddings: np.ndarray, 
                        labels: np.ndarray,
                        model_names: list,
                        full_model_list: list = None,
                        plot_points: int = 2000,
                        add_labels: bool = True) -> None:
    """
    Plot UMAP embeddings with model labels and color coding.
    
    Args:
        embeddings: UMAP embeddings
        labels: Array of labels for each point
        model_names: List of model names to plot
        full_model_list: Complete list of all possible models (for consistent color mapping)
        plot_points: Number of points to plot per model
        add_labels: Whether to add model name labels
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
    
    # Get the unique labels in the data
    unique_labels = np.unique(labels)
    
    # Create a mapping from model names to their indices in the full list
    model_to_full_index = {model: i for i, model in enumerate(full_model_list)}
    
    # Select random points for plotting
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
    plt.title('UMAP Embeddings')
    plt.grid()
    plt.show() 