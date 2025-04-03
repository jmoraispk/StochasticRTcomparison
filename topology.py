"""UMAP-based visualization and analysis of channel model topologies.

This module provides functions for visualizing and analyzing the topological
structure of channel data using UMAP dimensionality reduction. It includes
utilities for plotting embeddings and analyzing model relationships.
"""

import time

from cuml import UMAP  # type: ignore

from data_gen import DataConfig, load_data_matrices, prepare_data_for_analysis
from topology_utils import plot_umap_embeddings

#%% UMAP Visualization of Channel Models
# This notebook demonstrates the use of UMAP for visualizing the topological structure
# of different channel models.

#%% Load and Prepare Data
# Configure data generation
config = DataConfig()

# Example usage
ch_models = ['UMa']
rt_scens = ['asu_campus_3p5', 'city_0_newyork_3p5', 'city_1_losangeles_3p5']
models = rt_scens + ch_models

# Load and prepare data
data_matrices = load_data_matrices(models, config)
data_real, labels = prepare_data_for_analysis(data_matrices, models)

#%% Compute UMAP Embeddings
# Compute UMAP embeddings
seed = 42
umap_model = UMAP(n_components=2, random_state=seed)

print("Starting UMAP fit_transform...")
start_time = time.time()
umap_embeddings = umap_model.fit_transform(data_real)
umap_time = time.time() - start_time
print(f"UMAP fit_transform took {umap_time:.2f} seconds")

#%% Visualize Results
# Plot results
plot_umap_embeddings(umap_embeddings, labels, models) 