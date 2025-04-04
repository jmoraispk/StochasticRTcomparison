"""UMAP-based visualization and analysis of channel model topologies.

This module provides functions for visualizing and analyzing the topological
structure of channel data using UMAP dimensionality reduction. It includes
utilities for plotting embeddings and analyzing model relationships.
"""

#%%
import sys
sys.path.insert(0, '/mnt/c/Users/jmora/Documents/GitHub/StochasticRTcomparison/')

import time
import numpy as np
from cuml import UMAP  # type: ignore
import matplotlib.pyplot as plt

from data_gen import DataConfig, load_data_matrices, prepare_umap_data, remove_outliers
from topology_utils import plot_umap_embeddings

#%% Parameters
# Channel generation parameters
cfg = DataConfig()
cfg.n_samples = 10_000
cfg.n_prbs = 50  # Number of PRBs to generate channels
cfg.n_rx = 1
cfg.n_tx = 32
cfg.snr = 50

# Channel models  : ['Rayleigh', 'CDL-x', 'TDL-x', 'UMa', 'UMi'] 
# TR 38.901, x is : ["A", "B", "C", "D", "E"] 
# ch_models = ['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 
#              'TDL-A', 'TDL-B', 'TDL-C', 'Rayleigh', 'UMa', 'UMi']
ch_models = ['UMa']

# Ray tracing scenarios
rt_scens = ['asu_campus_3p5']
models = rt_scens + ch_models

# UMAP parameters
cfg.x_points = int(1e4)  # Number of points to sample from each dataset (randomly)
cfg.plot_points = 1e6    # Number of points to plot from each dataset (randomly)
cfg.seed = 42
cfg.rt_uniform_steps = [2, 2]


#%% Load and Prepare Data
# Load and prepare data
data_matrices = load_data_matrices(models, cfg)
data_real, labels = prepare_umap_data(data_matrices, models, x_points=cfg.x_points)

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
plot_umap_embeddings(umap_embeddings, labels, models, title="UMAP Embeddings (All Data)")

#%% Visualize Results (without outliers)
embeddings_clean, labels_clean = remove_outliers(umap_embeddings, labels, threshold=2.0)
print(f"Removed {len(umap_embeddings) - len(embeddings_clean)} outliers")

plot_umap_embeddings(embeddings_clean, labels_clean, models, title="UMAP Embeddings (No Outliers)")

#%% Partial UMAP Fitting
# Choose which models to fit and transform based on names
fit_model = 'UMa'  # Model to fit on
transform_model = 'asu_campus_3p5'  # Model to transform

# Get indices for fitting and transforming based on model names
fit_model_idx = models.index(fit_model)
transform_model_idx = models.index(transform_model)

# Get data for fitting and transforming using labels
fit_indices = np.where(labels == fit_model_idx)[0]
transform_indices = np.where(labels == transform_model_idx)[0]

data_fit = data_real[fit_indices]
data_transform = data_real[transform_indices]
labels_fit = labels[fit_indices]
labels_transform = labels[transform_indices]

# Fit UMAP on first model
umap_partial = UMAP(n_components=2, random_state=seed)
umap_partial.fit(data_fit)

# Transform the second model
embedding_fit = umap_partial.transform(data_fit)
embedding_transform = umap_partial.transform(data_transform)

# Combine embeddings and labels for plotting
embedding_combined = np.vstack([embedding_fit, embedding_transform])
labels_combined = np.concatenate([labels_fit, labels_transform])

# Plot combined results
plot_umap_embeddings(embedding_combined, labels_combined, models, 
                    title="Partial UMAP: Combined Results",
                    full_model_list=models)

#%% Plot Only Fitted Points
# Plot only the fitted points (first model) without the transformed points
fitted_model_names = [models[fit_model_idx]]
plot_umap_embeddings(embedding_fit, labels_fit, fitted_model_names, 
                    title=f"Partial UMAP: {fit_model} (Fitted)",
                    full_model_list=models)

# Get axis limits from the first plot
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

# Plot only the transformed points (second model) without the fitted points
fitted_model_names = [models[transform_model_idx]]
plot_umap_embeddings(embedding_transform, labels_transform, fitted_model_names, 
                    title=f"Partial UMAP: {transform_model} (Transformed)",
                    full_model_list=models,
                    xlim=xlim,
                    ylim=ylim)
