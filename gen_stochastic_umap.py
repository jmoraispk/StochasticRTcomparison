"""Generate stochastic channel data and prepare it for UMAP visualization.

This module generates channel data using Sionna channel models and prepares it
for dimensionality reduction and visualization using UMAP.
"""

#%% STARTING

# Standard library imports
import gc
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from cuml import UMAP  # type: ignore
from tqdm import tqdm

# Local imports
from channel_gen_funcs import SionnaChannelGenerator
from test_05_utils import get_matrix_name

SUBCARRIERS_PER_PRB = 12  # Hz

def prepare_data_for_umap(data_matrices, x_points=5000, release_memory=True):
    """Prepare data matrices for UMAP by sampling and concatenating.
    
    Args:
        data_matrices: List of channel matrices
        x_points: Number of points to sample from each matrix
        release_memory: Whether to free memory after processing
    
    Returns:
        tuple: (data_real, labels) where data_real is the concatenated real data
               and labels are the matrix indices
    """
    labels = []
    data = []
    
    for i, matrix in enumerate(data_matrices):
        available_points = matrix.shape[0]
        all_idxs = np.arange(available_points)
        random_idxs = np.random.choice(all_idxs, size=min(available_points, x_points), replace=False)
        data.append(matrix[random_idxs].reshape(len(random_idxs), -1))
        n_points = len(data[-1])
        labels.append(np.ones(n_points) * i)
        print(f"Loaded {n_points} points for matrix {i}")
        
        if release_memory:
            del matrix
    
    # Concatenate all data
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Separate real and imaginary parts
    data_real = np.concatenate([np.real(data), np.imag(data)], axis=1)
    
    if release_memory:
        del data
    
    return data_real, labels

def plot_umap_embeddings(embeddings, labels, model_names, full_model_list=None, plot_points=2000):
    """Plot UMAP embeddings with labels and model names.
    
    Args:
        embeddings: UMAP embeddings
        labels: Array of labels for each point
        model_names: List of model names to plot
        full_model_list: Complete list of all possible models (for consistent color mapping)
                        If None, model_names is assumed to be the complete list
        plot_points: Maximum number of points to plot per class
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
    
    # Add labels at means
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
            
        mean_x = np.mean(embeddings[class_mask, 0])
        mean_y = np.mean(embeddings[class_mask, 1])
        
        plt.annotate(model, xy=(mean_x, mean_y),
                    xytext=(mean_x + 1, mean_y + 1),
                    color=colors[i], fontsize=9, weight='bold',
                    bbox=dict(facecolor='grey', alpha=0.7, edgecolor='none', pad=0.5),
                    arrowprops=dict(facecolor=colors[i], shrink=0.05, width=1, headwidth=5))
    
    # Colorbar
    tick_locs = np.arange(n_models)
    scatter.set_clim(-0.5, n_models - 0.5)
    cbar = plt.colorbar(scatter, label='Channel Model', ticks=tick_locs,
                       boundaries=np.arange(-0.5, n_models + 0.5), values=np.arange(n_models))
    cbar.set_ticklabels(model_names)
    
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Embeddings')
    plt.grid()
    print(f'x lims = {plt.gca().get_xlim()}')
    print(f'y lims = {plt.gca().get_xlim()}')
    plt.show()

def sample_ch(ch_gen, n_prbs: int, n_iter: int = 100, batch_size: int = 10,
              snr: float = 50, n_rx: int = 1, n_tx: int = 10):
    """Generate channel samples using the provided channel generator.
    
    Args:
        ch_gen: Channel generator instance
        n_prbs: Number of PRBs in the UE allocated bandwidth
        n_iter: Number of iterations to generate samples
        batch_size: Number of samples per iteration
        snr: Signal-to-noise ratio in dB
        n_rx: Number of receive antennas
        n_tx: Number of transmit antennas
    
    Returns:
        ndarray: Generated channel samples of shape (n_samples, n_rx, n_tx, n_sub)
    """
    n_samples = n_iter * batch_size
    n_sub = n_prbs * SUBCARRIERS_PER_PRB
    d = np.zeros((n_iter, batch_size, n_rx, n_tx, n_sub), dtype=complex)
    
    print(f'Generating channels for SNR: {snr} dB')
    for i in (pbar := tqdm(range(n_iter))):
        _, d[i] = ch_gen.gen_channel_jit(snr)
        pbar.set_description(f"Iteration {i+1}/{n_iter}")
    
    return d.reshape(n_samples, n_rx, n_tx, n_sub)


def find_class_outliers(embeddings, labels, std_threshold=10):
    """Find outliers for each class that are more than std_threshold standard deviations from the mean.
    
    Args:
        embeddings: UMAP embeddings of shape (n_samples, 2)
        labels: Array of labels for each point
        std_threshold: Number of standard deviations for outlier detection
        
    Returns:
        dict: Dictionary mapping class indices to arrays of outlier indices
    """
    outliers_by_class = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Get points for this class
        class_mask = labels == label
        class_points = embeddings[class_mask]
        
        # Calculate mean and standard deviation for this class
        mean = np.mean(class_points, axis=0)
        std = np.std(class_points, axis=0)
        
        # Calculate Mahalanobis distance for each point
        distances = np.sqrt(np.sum(((class_points - mean) / std) ** 2, axis=1))
        
        # Find outliers
        class_outlier_mask = distances > std_threshold
        class_outlier_indices = np.where(class_mask)[0][class_outlier_mask]
        
        if len(class_outlier_indices) > 0:
            outliers_by_class[int(label)] = class_outlier_indices
    
    return outliers_by_class


def print_outliers(outliers, models):
    print("\nOutlier indices by class:")
    for class_idx, outlier_indices in outliers.items():
        print(f"Class {class_idx} ({models[class_idx]}): {len(outlier_indices)} outliers")
        print(f"  Indices: {outlier_indices}")


def get_mask_no_outliers(embedding, outliers_dict):
    keep_mask = np.ones(len(embedding), dtype=bool)
    for class_idx, outlier_indices in outliers_dict.items():
        curr_mask = ~np.isin(np.arange(len(embedding)), outlier_indices)
        keep_mask &= curr_mask
    return keep_mask
    

#%% Parameters
# Channel generation parameters
n_iter = 3000
batch_size = 10
n_prbs = 200  # Number of PRBs to generate channels
n_rx = 1
n_tx = 32
snr = 50

# Channel models  : ['Rayleigh', 'CDL-x', 'TDL-x', 'UMa', 'UMi'] 
# TR 38.901, x is : ["A", "B", "C", "D", "E"] 
# ch_models = ['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 
#              'TDL-A', 'TDL-B', 'TDL-C', 'Rayleigh', 'UMa', 'UMi']

ch_models = ['UMa']

# Frequency selection parameters
max_n_prbs = 50  # Maximum number of PRBs in the UE allocated bandwidth
freq_selection = np.round(np.linspace(0, max_n_prbs * SUBCARRIERS_PER_PRB, max_n_prbs)).astype(int)

# UMAP parameters
x_points = int(1e8)  # Number of points to sample from each matrix

seed = 40

#%% Save generated channels

# Save parameters
save_folder = 'stochastic_data'
os.makedirs(save_folder, exist_ok=True)

for ch_model in ch_models:
    ch_gen = SionnaChannelGenerator(n_prbs, ch_model, batch_size, n_rx=n_rx, n_tx=n_tx)
    ch_data = sample_ch(ch_gen, n_prbs, n_iter, batch_size, snr=snr, n_rx=n_rx, n_tx=n_tx)
    fname = get_matrix_name(ch_model, batch_size * n_iter, n_prbs, n_tx, n_rx)
    np.save(os.path.join(save_folder, fname), ch_data)

#%% Data generation and collection for UMAP
ch_gens = [SionnaChannelGenerator(n_prbs, ch_model, batch_size, n_rx, n_tx, seed)
           for ch_model in ch_models]

# Initialize empty list to store data from all generators
all_ch_data = []

# Generate and collect data from all channel generators
for ch_gen in ch_gens:
    ch_data = sample_ch(ch_gen, n_prbs, n_iter, batch_size, snr, n_rx, n_tx)
    ch_data_t = ch_data[:, :, :, freq_selection].astype(np.complex64)
    all_ch_data.append(ch_data_t)
    del ch_data
    gc.collect()

#%% Data preparation and UMAP processing
data_real, labels = prepare_data_for_umap(all_ch_data, x_points=x_points)

print("Running UMAP dimensionality reduction...")
umap = UMAP(n_components=2, random_state=42)
embedding = umap.fit_transform(data_real)

#%% Visualization

# Find and print outliers for each class
outliers_per_class = find_class_outliers(embedding, labels)

print(outliers_per_class, ch_models)

# plot_umap_embeddings(embedding, labels, ch_models, plot_points=int(1e6))
keep_mask = ~np.isin(np.arange(len(embedding)), outliers_per_class[0])
plot_umap_embeddings(embedding[keep_mask], labels[keep_mask], ch_models, full_model_list=ch_models, plot_points=int(1e6))

#%% Partial UMAP fitting & Plot
# Try UMAP with partial fitting on different datasets
print("\nTrying UMAP with partial fitting on different datasets...")

# Choose which model to fit on and which to transform
fit_model_idx = 0  # First model for fitting
transform_model_idx = 1  # Second model for transforming

# Get indices for fitting and transforming based on labels
fit_indices = np.where(labels == fit_model_idx)[0]
transform_indices = np.where(labels == transform_model_idx)[0]

# Get data for fitting and transforming using indices
data_fit, data_transform = data_real[fit_indices], data_real[transform_indices]
labels_fit, labels_transform = labels[fit_indices], labels[transform_indices]

# Fit UMAP on first model
umap_partial = UMAP(n_components=2, random_state=42)
umap_partial.fit(data_fit)

# Transform the second model
embedding_fit = umap_partial.transform(data_fit)
embedding_transform = umap_partial.transform(data_transform)

# Combine embeddings and labels for plotting
embedding_combined = np.vstack([embedding_fit, embedding_transform])
labels_combined = np.concatenate([labels_fit, labels_transform])

# Use the existing plotting function
plot_umap_embeddings(embedding_combined, labels_combined, ch_models, full_model_list=ch_models, plot_points=int(1e6))

#%% Partial UMAP fitting - Plot only the fit (not transformed)

# Plot only the fitted points (first model) without the transformed points
# Create a subset of model names that matches the data we're plotting
fitted_model_names = [ch_models[fit_model_idx]]
plot_umap_embeddings(embedding_fit, labels_fit, fitted_model_names, full_model_list=ch_models, plot_points=int(1e6))

# HERE:
# Try UMAP on UMa + adding CDL-A
# Try UMAP on CDL-A + adding UMa

#%%

import deepmimo as dm

relevant_mats = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 
                 'power', 'phase', 'delay', 'rx_pos', 'tx_pos']
load_params = dict(tx_sets=[1], rx_sets=[0], matrices=relevant_mats)

dataset = dm.load('asu_campus_3p5', **load_params)

ch_params = dm.ChannelGenParameters()
ch_params.ofdm.bandwidth = 15e3 * n_prbs * 12
ch_params.ofdm.num_subcarriers = n_prbs * 12
ch_params.ofdm.selected_subcarriers = freq_selection
ch_params.bs_antenna.shape = np.array([n_tx, 1])
ch_params.ue_antenna.shape = np.array([n_rx, 1])
# ch_params.bs_antenna.rotation = np.array([0, 0, -135])
ch_params.ue_antenna.rotation = np.array([0, 0, 0])

# Reduce dataset size with uniform sampling
dataset_u = dataset.subset(dataset.get_uniform_idxs([2,1]))

# Consider only active users for redundancy reduction
dataset_t = dataset.subset(dataset_u.get_active_idxs())

rt_ch = dataset_t.compute_channels(ch_params)

# IMPORTANT NOTE ABOUT DIMENSIONS:
# if we want to use several rx and several tx antennas, we need to make
# sure they are along the same dimensions in the RT and Stochastic datasets
# CURRENTLY THEY ARE NOT. 

if len(all_ch_data) == 1:
    all_ch_data.append(rt_ch)
else:
    all_ch_data[1] = rt_ch

print(f'rt_ch.shape = {rt_ch.shape}')

#%%
x_points = int(30e3) 

# Transform UMAP
data_real, labels = prepare_data_for_umap(all_ch_data, x_points=x_points)

print("Running UMAP dimensionality reduction...")
umap = UMAP(n_components=2, random_state=42)
embedding = umap.fit_transform(data_real)

models = ch_models + ['asu']

#%% Plot
plot_umap_embeddings(embedding, labels, models, plot_points=int(1e6))

#%% Plot without outliers
outliers_per_class = find_class_outliers(embedding, labels, std_threshold=1)
print_outliers(outliers_per_class, models)

keep_mask = get_mask_no_outliers(embedding, outliers_per_class)
plot_umap_embeddings(embedding[keep_mask], labels[keep_mask], models, plot_points=int(1e6))

#%% 

# Plot only 
class_sel = 'UMa'
class_mask = labels[keep_mask] == models.index(class_sel)
plot_umap_embeddings(embedding[keep_mask][class_mask], 
                     labels[keep_mask][class_mask], [class_sel],
                     full_model_list=models, plot_points=int(1e6))


# Try UMa vs RT
# Next: try UMAP on UMa + adding RT

# ----
# Try RT with different parameters

