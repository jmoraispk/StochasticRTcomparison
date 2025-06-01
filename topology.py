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
    # normalize = False,
    normalize = 'dataset-mean-var-complex',
    # normalize = 'dataset-mean-precise'
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
ch_models = ['CDL-D', 'UMa']

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

# UMAP with flipped plot order (better visualization of what's inside what)
# plot_umap_embeddings(umap_embeddings, labels, [models[2], models[0], models[1]], 
#                      full_model_list=models, plot_points=cfg.plot_points,
#                      title="UMAP Embeddings (All Data) - plot order flipped")

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

single_model = 'CDL-D'
model_indices = np.where(labels == models.index(single_model))[0]
single_model_embeddings_e = umap_embeddings[model_indices]
plot_umap_heatmap(single_model_embeddings_e, single_model, (xmin, xmax), (ymin, ymax))


#%% [Part 1] Sampling method using RT as baseline for UMa/UMi parameterization

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
data_matrices = {}

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

dataset_t = dataset_a.subset(np.arange(10_000))
print(f"After final sampling: {dataset_t.n_ue} UEs")

mat_rt = dataset_t.compute_channels(ch_params)

#%% [Part 2] Generate UMa/UMi data

def process_batch(dm_dataset, batch_idxs, los_status, config, stochastic_model):
    """Process a single batch of users with specified LoS status.
    
    Args:
        dm_dataset: DeepMIMO dataset containing user positions and orientations
        batch_idxs: Indices of users to process in this batch
        los_status: 1 for LoS, 0 for NLoS
        config: Configuration object
        stochastic_model: Name of the stochastic model to use
        
    Returns:
        Channel data for the batch
    """
    # Base station position and orientation
    bs_pos = dm_dataset.tx_pos.reshape((1, 1, 3))
    bs_ori = dm_dataset.tx_ori.reshape((1, 1, 3)) * np.pi / 180  # to radians
    
    # Process users
    rx_pos = dm_dataset.rx_pos[batch_idxs].reshape((1, -1, 3))
    rx_ori = np.zeros((1, len(batch_idxs), 3))
    
    topology = TopologyConfig(
        ut_loc=tf.cast(rx_pos, tf.float32),
        bs_loc=tf.cast(bs_pos, tf.float32),
        ut_orientations=tf.cast(rx_ori, tf.float32),
        bs_orientations=tf.cast(bs_ori, tf.float32),
        ut_velocities=tf.cast(rx_ori, tf.float32),
        in_state=tf.cast(np.zeros((1, len(batch_idxs)), dtype=bool), tf.bool),
        los=bool(los_status))
    
    # Create channel generator
    ch_gen_params = dict(num_prbs=config.n_prbs, channel_name=stochastic_model, 
                        batch_size=1, n_rx=config.n_rx, n_tx=config.n_tx,
                        normalize=False, seed=config.seed, n_ue=len(batch_idxs))
    
    ch_gen = SionnaChannelGenerator(**ch_gen_params, topology=topology)
    
    # Sample data
    ch_data = sample_ch(ch_gen, config.n_prbs, 1, len(batch_idxs), 
                        config.snr, config.n_rx, config.n_tx)
    
    return ch_data

# 2.1- Get LoS & NLoS indices
los_idxs = np.where(dataset_t.los == 1)[0]
nlos_idxs = np.where(dataset_t.los == 0)[0]

# 2.2- Process users in batches
batch_size = 200
ch_data_list = []

# Process all LoS users first, then all NLoS users
for los_status, idxs in [(1, los_idxs), (0, nlos_idxs)]:
    # Create batches of indices
    batches = [idxs[i:i + batch_size] for i in range(0, len(idxs), batch_size)]
    status_str = "LoS" if los_status == 1 else "NLoS"
    
    # Process each batch
    for i, batch in enumerate(batches):
        print(f"Processing {status_str} batch {i + 1}/{len(batches)}...")
        ch_data = process_batch(dataset_t, batch, los_status, config, stochastic_model)
        ch_data_list.append(ch_data)

# 2.3- Concatenate results
ch_data = np.concatenate(ch_data_list, axis=0)

# 2.4- Process final matrices
mat_stochastic = ch_data[:, :, :, config.freq_selection].astype(np.complex64)
print(f"Generated {mat_stochastic.shape[0]} samples for {stochastic_model}")

#%% [Part 3] Normalize data and prepare for UMAP

# 2.5- Normalize data
data_matrices[rt_model] = normalize_data(mat_rt, mode=cfg.normalize)
data_matrices[stochastic_model] = normalize_data(mat_stochastic, mode=cfg.normalize)

data_real, labels = prepare_umap_data(data_matrices, models, cfg.x_points, cfg.seed)