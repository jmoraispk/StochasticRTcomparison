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
from topology_utils import (plot_umap_embeddings, plot_umap_3d_histogram, 
                            plot_amplitude_distribution, plot_umap_heatmap)

#%% Parameters
# Channel generation parameters
cfg = DataConfig(
    n_samples = 10_000,
    n_prbs = 20,
    n_rx = 1,
    n_tx = 32,
    snr = 50,
    batch_size = 10, 
    normalize = 'dataset-mean-var-complex'
    # normalize = 'dataset-mean-var'
    # per 'datapoint', 'dataset-minmax', 
    # 'dataset-mean-around-order', 'dataset-mean-precise',
    # 'dataset-mean-var',
    # 'dataset-mean-complex', 'dataset-mean-var-complex'
)

# UMAP parameters
cfg.x_points = cfg.n_samples #int(2e5)  # Number of points to sample from each dataset (randomly)
cfg.plot_points = cfg.n_samples # No. points to plot from each dataset (randomly). None = all points.
cfg.seed = 42  # Set to None to keep random
cfg.rt_uniform_steps = [2, 2]

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
data_real, labels = prepare_umap_data(data_matrices, models, cfg.x_points, cfg.seed)

#%% Compute UMAP Embeddings

umap_model = UMAP(n_components=2, random_state=cfg.seed)

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

xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()
# plt.xlim((-7, 5))
# plt.ylim((-9, 7))

#%% Visualize Single Model (free label scale)
single_model = 'asu_campus_3p5'
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

#%% Visualize Single Model (keep label scale)

other_model = [model for model in models if model != single_model][0]

other_model_idxs = np.where(labels == other_model_idx)[0][0]
model_indices[0] = other_model_idxs
# Select a single model to visualize + 1 sample of the other models for maintaining the same scale

single_model_embeddings_e = umap_embeddings[model_indices]
single_model_labels_e = labels[model_indices]

plot_umap_embeddings(single_model_embeddings_e, single_model_labels_e, models,
                     plot_points=cfg.plot_points,
                     title=f"UMAP Embeddings ({single_model} Only)")
plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))

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

# %matplotlib QtAgg
# %matplotlib inline
plot_umap_3d_histogram(umap_3d, labels_3d, models, 
                       n_bins=50,
                       plot_points=cfg.plot_points,
                       title="UMAP 3D Density Visualization",
                       alpha=0.4,
                       normalize=True,
                       density_threshold=1.0)  # Only show densities above 1%

#%% Amplitude distributions

# Transform real data into complex data
real_data_shape = (data_real.shape[0], data_real.shape[1]//2) 
data_complex = data_real.view(np.complex64).reshape(real_data_shape)

# Calculate amplitudes for each point
amplitude_data = np.sqrt(np.sum(np.abs(data_complex)**2, axis=1))

plot_amplitude_distribution(amplitude_data, labels, models, 
                            title="Amplitude Distribution by Model",
                            density=True)

#%% Raw Data plots: Plot real and imaginary parts of the data

plt.figure(dpi=150)
n = 1000
asu_data = data_matrices['asu_campus_3p5'][:n]
uma_data = data_matrices['UMa'           ][:n]
# plt.scatter(asu_data.real, asu_data.imag, s=1, label='asu', color='tab:blue')
plt.scatter(uma_data.real, uma_data.imag, s=1, label='uma', color='tab:orange')
plt.legend(loc='upper right')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid()
plt.show()

#%% Check amplitudes directly

amps1 = np.sqrt(np.sum(np.abs(data_matrices['asu_campus_3p5'])**2, axis=(1,2,3)))
amps2 = np.sqrt(np.sum(np.abs(data_matrices['UMa'           ])**2, axis=(1,2,3)))

plt.figure(dpi=150)
plt.scatter(range(len(amps1)), amps1, label='asu', s=1)
plt.scatter(range(len(amps1), len(amps1) + len(amps2)), amps2, label='uma', s=1)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

print(f'asu mean norm: {np.mean(amps1)}')
print(f'uma mean norm: {np.mean(amps2)}')

print(f"asu mean amplitude: {np.mean(np.abs(data_matrices['asu_campus_3p5']))}")
print(f"uma mean amplitude: {np.mean(np.abs(data_matrices['UMa'           ]))}")

#%% Plot 2D Heatmaps of UMAP Embeddings sample density

single_model = 'UMa'
model_indices = np.where(labels == models.index(single_model))[0]
single_model_embeddings_e = umap_embeddings[model_indices]
plot_umap_heatmap(single_model_embeddings_e, single_model, (xmin, xmax), (ymin, ymax))

single_model = 'asu_campus_3p5'
model_indices = np.where(labels == models.index(single_model))[0]
single_model_embeddings_e = umap_embeddings[model_indices]
plot_umap_heatmap(single_model_embeddings_e, single_model, (xmin, xmax), (ymin, ymax))


#%% Sampling method using RT as baseline for UMa/UMi parameterization

# TODO: put this inside data_gen.py as a load_based_on_rt_model

# - when moving this into data_gen.py, isolate the RT-based data gen so it can be
#   used for other functions as well. 
# - load_based_on_rt_model should take a single RT model and a single stochastic model (UMa or UMi)
#   and return a dictionary of a matrix for the stochastic model (or both?)


import deepmimo as dm
from sionna_ch_gen import SionnaChannelGenerator, TopologyConfig
from data_gen import normalize_data, sample_ch
import tensorflow as tf

rt_model = 'asu_campus_3p5'
stochastic_model = 'UMa'
config = cfg
steps = cfg.rt_uniform_steps
SUBCARRIERS_PER_PRB = 12

# 1- Load RT data
print(f"Loading ray tracing data for {rt_model}...")
            
# The tx-ID may be hinted by adding '!1' or '!2', to model name
possible_tx_id = rt_model.split('!')[-1]            
tx_id = int(possible_tx_id) if possible_tx_id.isdigit() else 1

load_params = dict(tx_sets=[tx_id], rx_sets=[0], matrices=config.relevant_mats)
dataset = dm.load(rt_model, **load_params)

# Adjust uniform steps based on dataset size
print(f"Using uniform sampling steps {steps} for {dataset.n_ue} UEs")

ch_params = dm.ChannelParameters()
ch_params.ofdm.bandwidth = 15e3 * config.n_prbs * SUBCARRIERS_PER_PRB
ch_params.ofdm.num_subcarriers = config.n_prbs * SUBCARRIERS_PER_PRB
ch_params.ofdm.selected_subcarriers = config.freq_selection
ch_params.bs_antenna.shape = np.array([config.n_tx, 1])
ch_params.ue_antenna.shape = np.array([config.n_rx, 1])
ch_params.bs_antenna.rotation = np.array([0, 0, -135])

# Reduce dataset size with uniform sampling
dataset_u = dataset.subset(dataset.get_uniform_idxs(steps))
print(f"After uniform sampling: {dataset_u.n_ue} UEs")

# Consider only active users for redundancy reduction
dataset_a = dataset_u.subset(dataset_u.get_active_idxs())
print(f"After active user filtering: {dataset_a.n_ue} UEs")

#mat = dataset_t.compute_channels(ch_params)

# Sample RT data
usr_idxs = 0

# TODO: try to do all sampling at once. If doesn't work, distribute in batches.
dataset_t = dataset_a.subset(np.arange(10_000))
print(f"After final sampling: {dataset_t.n_ue} UEs")

#%%
# 2- Generate UMa/UMi data

# 2.1- Create LoS & NLoS Topologies
b_size = 1 # batch size
bs_pos = dataset_t.tx_pos.reshape((b_size, 1, 3))
bs_ori = dataset_t.tx_ori.reshape((b_size, 1, 3)) * np.pi / 180 # to radians

# LoS
los_idxs = np.where(dataset_t.los == 1)[0]
rx_pos_los = dataset_t.rx_pos[los_idxs].reshape((b_size, -1, 3))  # Reshape to [batch, num_ut, 3]
rx_ori_los = np.zeros((b_size, len(los_idxs), 3))  # same as velocities

topology_los = TopologyConfig(
    ut_loc=tf.cast(rx_pos_los, tf.float32),
    bs_loc=tf.cast(bs_pos, tf.float32),
    ut_orientations=tf.cast(rx_ori_los, tf.float32),
    bs_orientations=tf.cast(bs_ori, tf.float32),
    ut_velocities=tf.cast(rx_ori_los, tf.float32),
    in_state=tf.cast(np.zeros((b_size, len(los_idxs)), dtype=bool), tf.bool),
    los=True)

# NLoS
nlos_idxs = np.where(dataset_t.los == 0)[0]
rx_pos_nlos = dataset_t.rx_pos[nlos_idxs].reshape((b_size, -1, 3))  # Reshape to [batch, num_ut, 3]
rx_ori_nlos = np.zeros((b_size, len(nlos_idxs), 3))  # same as velocities

topology_nlos = TopologyConfig(
    ut_loc=tf.cast(rx_pos_nlos, tf.float32),
    bs_loc=tf.cast(bs_pos, tf.float32),
    ut_orientations=tf.cast(rx_ori_nlos, tf.float32),
    bs_orientations=tf.cast(bs_ori, tf.float32),
    ut_velocities=tf.cast(rx_ori_nlos, tf.float32),
    in_state=tf.cast(np.zeros((b_size, len(nlos_idxs)), dtype=bool), tf.bool),
    los=False)

# 2.2- Create LoS & NLos channel generators
ch_gen_params = dict(num_prbs=config.n_prbs, channel_name=stochastic_model, 
                     batch_size=1, n_rx=config.n_rx, n_tx=config.n_tx,
                     normalize=False, seed=config.seed)

print(f"Creating channel generator for {stochastic_model} LoS...")
los_ch_gen = SionnaChannelGenerator(**ch_gen_params, topology=topology_los)

print(f"Creating channel generator for {stochastic_model} NLos...")
nlos_ch_gen = SionnaChannelGenerator(**ch_gen_params, topology=topology_nlos)

# 2.3- Sample data for LoS & NLoS (batch size = 1, since we must sample all users at once)
print(f"Sampling {len(los_idxs)} LoS samples...")
los_ch_data = sample_ch(los_ch_gen, config.n_prbs, 1, len(los_idxs), 
                        config.snr, config.n_rx, config.n_tx)

print(f"Sampling {len(nlos_idxs)} NLos samples...")
nlos_ch_data = sample_ch(nlos_ch_gen, config.n_prbs, 1, len(nlos_idxs), 
                        config.snr, config.n_rx, config.n_tx)

los_mat = los_ch_data[:, :, :, config.freq_selection].astype(np.complex64)
nlos_mat = nlos_ch_data[:, :, :, config.freq_selection].astype(np.complex64)
mat = np.concatenate([los_mat, nlos_mat], axis=0)
print(f"Generated {mat.shape[0]} samples for {rt_model}")

# 2.4- Normalize data
data_matrices[rt_model] = normalize_data(mat, mode=config.normalize)

