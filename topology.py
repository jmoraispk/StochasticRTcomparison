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
from topology_utils import plot_umap_embeddings, plot_umap_3d_histogram

#%% Parameters
# Channel generation parameters
cfg = DataConfig(
    n_samples = 100_000,  # For Stochastic models
    n_prbs = 20,
    n_rx = 1,
    n_tx = 32,
    snr = 50,
    normalize = True
)

# UMAP parameters
cfg.x_points = cfg.n_samples #int(2e5)  # Number of points to sample from each dataset (randomly)
cfg.plot_points = cfg.n_samples # No. points to plot from each dataset (randomly). None = all points.
cfg.seed = 42  # TODO: put False to keep random
cfg.rt_uniform_steps = [1, 1]

# Channel models  : ['Rayleigh', 'CDL-x', 'TDL-x', 'UMa', 'UMi'] 
# TR 38.901, x is : ["A", "B", "C", "D", "E"] 
# ch_models = ['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 
#              'TDL-A', 'TDL-B', 'TDL-C', 'Rayleigh', 'UMa', 'UMi']
ch_models = ['UMa']

# Ray tracing scenarios
rt_scens = ['asu_campus_3p5'] # add '!1' at the end to cue tx-id = 1
# rt_scens = ['asu_campus_3p5',
#             'city_0_newyork_3p5!1', 'city_0_newyork_3p5!2', 'city_0_newyork_3p5!3',
#             'city_1_losangeles_3p5!1', 'city_1_losangeles_3p5!2', 'city_1_losangeles_3p5!3',
#             'city_2_chicago_3p5!1', 'city_2_chicago_3p5!2', 'city_2_chicago_3p5!3']
models = rt_scens + ch_models

#%% Load and Prepare Data

data_matrices = load_data_matrices(models, cfg)
data_real, labels = prepare_umap_data(data_matrices, models, x_points=cfg.x_points)

#%% Compute UMAP Embeddings

umap_model = UMAP(n_components=2)#, random_state=cfg.seed)

print("Starting UMAP fit_transform...")
start_time = time.time()
umap_embeddings = umap_model.fit_transform(data_real)
umap_time = time.time() - start_time
print(f"UMAP fit_transform took {umap_time:.2f} seconds")

#%% Visualize Results
plot_umap_embeddings(umap_embeddings, labels, models, plot_points=cfg.plot_points,
                     title="UMAP Embeddings (All Data)")

# UMAP with flipped plot order (better visualization of what's inside what)
plot_umap_embeddings(umap_embeddings, labels, models[::-1], 
                     full_model_list=models, plot_points=cfg.plot_points,
                     title="UMAP Embeddings (All Data) - plot order flipped")

# plt.xlim((-11, 8))
# plt.ylim((-7, 7))

#%% Visualize Single Model
single_model = 'UMa'
single_model_idx = models.index(single_model)
model_indices = np.where(labels == single_model_idx)[0]

# Get embeddings and labels for just this model
single_model_embeddings = umap_embeddings[model_indices]
single_model_labels = labels[model_indices]

# Plot just this model's embeddings
plot_umap_embeddings(single_model_embeddings, single_model_labels, [single_model],
                     plot_points=cfg.plot_points,
                     title=f"UMAP Embeddings ({single_model} Only)",
                     full_model_list=models)

#%% Visualize Single Model (with other model for keeping the labels behaved)

other_model_idxs = np.where(labels == models.index('asu_campus_3p5'))[0][0]
model_indices[0] = other_model_idxs
# Select a single model to visualize + 1 sample of the other models for maintaining the same scale

single_model_embeddings_e = umap_embeddings[model_indices]
single_model_labels_e = labels[model_indices]

plot_umap_embeddings(single_model_embeddings_e, single_model_labels_e, models,
                     plot_points=cfg.plot_points,
                     title=f"UMAP Embeddings ({single_model} Only)")
plt.xlim((-11, 8))
plt.ylim((-7, 7))

#%% Visualize Results (without outliers)
embeddings_clean, labels_clean = remove_outliers(umap_embeddings, labels, threshold=1.5)
print(f"Removed {len(umap_embeddings) - len(embeddings_clean)} outliers")

plot_umap_embeddings(embeddings_clean, labels_clean, models, plot_points=cfg.plot_points,
                     title="UMAP Embeddings (No Outliers)")


# UMAP with flipped plot order (better visualization of what's inside what)
plot_umap_embeddings(embeddings_clean, labels_clean, models[::-1], 
                     full_model_list=models, plot_points=cfg.plot_points,
                     title="UMAP Embeddings (No Outliers) - plot order flipped")


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
umap_partial = UMAP(n_components=2, random_state=cfg.seed)
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

#%% 3D Densities

umap_3d, labels_3d = umap_embeddings, labels
# umap_3d, labels_3d = embeddings_clean, labels_clean

%matplotlib QtAgg
# %matplotlib inline
plot_umap_3d_histogram(umap_3d, labels_3d, models, 
                       n_bins=50,
                       plot_points=cfg.plot_points,
                       title="UMAP 3D Density Visualization",
                       alpha=0.4,
                       normalize=True,
                       density_threshold=1.0)  # Only show densities above 1%
