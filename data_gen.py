"""Core data generation and preparation functions for channel model comparison.

This module provides functions for loading, preparing, and analyzing channel data
from both stochastic and ray tracing models. It includes utilities for data
configuration, matrix loading, outlier detection, and data preparation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import deepmimo as dm  # type: ignore
import numpy as np
from tqdm import tqdm
from sionna_ch_gen import SionnaChannelGenerator

# Constants
SUBCARRIERS_PER_PRB = 12  # Hz

# Models that are stochastic are ray tracing scenarios
STOCHASTIC_MODELS = ['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 
                     'TDL-A', 'TDL-B', 'TDL-C', 'Rayleigh', 'UMa', 'UMi']
RT_MODELS = ['asu_campus_3p5', 'city_0_newyork_3p5', 'city_1_losangeles_3p5',
            'city_2_chicago_3p5', 'city_3_houston_3p5', 'city_4_phoenix_3p5',
            'city_5_philadelphia_3p5', 'city_6_miami_3p5', 'city_7_sandiego_3p5',
            'city_8_dallas_3p5', 'city_9_sanfrancisco_3p5', 'city_10_austin_3p5',
            'city_11_santaclara_3p5', 'city_12_fortworth_3p5', 'city_13_columbus_3p5',
            'city_14_charlotte_3p5', 'city_15_indianapolis_3p5', 'city_16_sanfrancisco_3p5',
            'city_17_seattle_3p5', 'city_18_denver_3p5', 'city_19_oklahoma_3p5',]

@dataclass
class RTConfig:
    """Configuration for a single ray tracing scenario"""
    uniform_steps: List[int] = None  # Steps for uniform sampling
    active_only: bool = True  # Whether to use only active users

@dataclass
class DataConfig:
    """Configuration for data generation"""
    n_prbs: int = 50
    n_tx: int = 10
    n_rx: int = 1
    n_samples: int = 10_000
    batch_size: int = 10
    data_folder: str = 'stochastic_data'
    relevant_mats: List[str] = None
    x_points: int = int(1e8)  # Number of points to sample from each matrix
    plot_points: int = 1000  # Number of points to plot
    seed: int = 40
    normalize: bool = True  # Whether to normalize the channels
    snr: int = 50
    freq_selection: np.ndarray = None  # Will be computed in __post_init__
    rt_uniform_steps: List[int] = None  # Steps for uniform sampling in ray tracing

    def __post_init__(self):
        """Initialize derived parameters after instance creation."""
        if self.relevant_mats is None:
            self.relevant_mats = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 
                                  'power', 'phase', 'delay', 'rx_pos', 'tx_pos']
        
        # Compute frequency selection (one subcarrier per PRB)
        if self.freq_selection is None:
            self.freq_selection = np.arange(0, self.n_prbs * SUBCARRIERS_PER_PRB, SUBCARRIERS_PER_PRB)
            
        # Default uniform steps based on dataset size
        if self.rt_uniform_steps is None:
            self.rt_uniform_steps = [3, 3]  # Default for large datasets

def load_data_matrices(models: List[str], config: DataConfig) -> Dict[str, np.ndarray]:
    """Load data matrices for specified models.
    
    Args:
        models: List of model names to load
        config: DataConfig instance with generation parameters
        
    Returns:
        Dictionary mapping model names to their data matrices
    """
    data_matrices = {}

    for model in models:
        model_name = model.split('!')[0]
        print(f"\nProcessing {model}...")
        
        if model_name in STOCHASTIC_MODELS:
            print(f"Generating stochastic data for {model}...")
            ch_gen = SionnaChannelGenerator(num_prbs=config.n_prbs,
                                            channel_name=model,
                                            batch_size=config.batch_size,
                                            n_rx=config.n_rx,
                                            n_tx=config.n_tx,
                                            normalize=False, #config.normalize, #3
                                            seed=config.seed)
            ch_data = sample_ch(ch_gen, config.n_prbs, config.n_samples // config.batch_size, 
                                config.batch_size, config.snr, config.n_rx, config.n_tx)
            mat = ch_data[:, :, :, config.freq_selection].astype(np.complex64)
            
        elif model_name in RT_MODELS:
            print(f"Loading ray tracing data for {model}...")
            
            # The tx-ID may be hinted by adding '!1' or '!2', to model name
            possible_tx_id = model.split('!')[-1]            
            tx_id = int(possible_tx_id) if possible_tx_id.isdigit() else 1

            load_params = dict(tx_sets=[tx_id], rx_sets=[0], matrices=config.relevant_mats)
            dataset = dm.load(model_name, **load_params)
            
            # Adjust uniform steps based on dataset size
            steps = config.rt_uniform_steps
            if steps is None:  # Only use dataset size logic if steps not configured
                if dataset.n_ue > 100_000:
                    steps = [3, 3]
                elif dataset.n_ue > 10_000:
                    steps = [2, 2]
                else:
                    steps = [1, 1]
            print(f"Using uniform sampling steps {steps} for {dataset.n_ue} UEs")
            
            ch_params = dm.ChannelGenParameters()
            ch_params.ofdm.bandwidth = 15e3 * config.n_prbs * SUBCARRIERS_PER_PRB
            ch_params.ofdm.num_subcarriers = config.n_prbs * SUBCARRIERS_PER_PRB
            ch_params.ofdm.selected_subcarriers = config.freq_selection
            ch_params.bs_antenna.shape = np.array([config.n_tx, 1])
            ch_params.ue_antenna.shape = np.array([config.n_rx, 1])
            ch_params.ue_antenna.rotation = np.array([0, 0, 0])

            # Reduce dataset size with uniform sampling
            dataset_u = dataset.subset(dataset.get_uniform_idxs(steps))
            print(f"After uniform sampling: {dataset_u.n_ue} UEs")

            # Consider only active users for redundancy reduction
            dataset_t = dataset.subset(dataset_u.get_active_idxs())
            print(f"After active user filtering: {dataset_t.n_ue} UEs")

            mat = dataset_t.compute_channels(ch_params)
        else:
            raise Exception(f'Model {model} not recognized.')

        print(f"Generated {mat.shape[0]} samples for {model}")

        # Normalize
        if config.normalize:
            if config.normalize == 'datapoint':
                ch_norms = np.sqrt(np.sum(np.abs(mat)**2, axis=(1,2,3), keepdims=True))
                non_zero_ues = np.where(ch_norms[:,0,0,0] > 0)[0]
                
                # if non zero is diff from the actual number of UEs, print warning
                if non_zero_ues.shape[0] != mat.shape[0]:
                    print(f"Warning: {model} has {data_matrices[model].shape[0]} UEs, "
                        f"but {non_zero_ues.shape[0]} non-zero UEs")
                data_matrices[model] = mat[non_zero_ues] / ch_norms[non_zero_ues]
        else:
            data_matrices[model] = mat
    
    return data_matrices

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

def prepare_umap_data(data_matrices: dict, model_names: list, x_points: int = 5000, 
                      seed: Optional[int] = None, release_memory: bool = True, 
                      convert_to_real: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data matrices for UMAP analysis by sampling and concatenating.
    
    Args:
        data_matrices: Dictionary of data matrices
        model_names: List of model names
        x_points: Number of points to sample from each matrix
        release_memory: Whether to free memory after processing
    
    Returns:
        Tuple of (data_real, labels) where data_real is the concatenated real data
        and labels are the matrix indices
    """
    labels = []
    data = []
    
    if seed:
        np.random.seed(seed)
    
    for i, model in enumerate(model_names):
        matrix = data_matrices[model]
        available_points = matrix.shape[0]
        all_idxs = np.arange(available_points)
        random_idxs = np.random.choice(all_idxs, size=min(available_points, x_points), replace=False)
        data.append(matrix[random_idxs].reshape(len(random_idxs), -1))
        n_points = len(data[-1])
        labels.append(np.ones(n_points) * i)
        print(f"Loaded {n_points} points for {model}")
        
        if release_memory:
            del matrix
    
    # Concatenate all data
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if convert_to_real:
        data = data.view(np.float32).reshape(data.shape[0], -1)
    
    return data, labels

def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in the data using z-score method.
    
    Args:
        data: Input data matrix
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean array indicating which samples are outliers
    """
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return np.any(z_scores > threshold, axis=1)

def remove_outliers(data: np.ndarray, labels: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from the data and corresponding labels.
    
    Args:
        data: Input data matrix
        labels: Corresponding labels
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of (cleaned_data, cleaned_labels)
    """
    outlier_mask = detect_outliers(data, threshold)
    return data[~outlier_mask], labels[~outlier_mask]

def get_matrix_name(ch_model: str, n_samples: int, n_prb: int, n_tx: int = 1, n_rx: int = 1) -> str:
    """
    Generate standardized filename for data matrices.
    
    Args:
        ch_model: Channel model name
        n_samples: Number of samples
        n_prb: Number of PRBs
        n_tx: Number of TX antennas
        n_rx: Number of RX antennas
        
    Returns:
        Standardized filename string
    """
    if n_tx == 1 and n_rx == 1:
        return f'data_{ch_model}_samples_{n_samples}_n-prb_{n_prb}.npy'
    else:
        return f'data_{ch_model}_samples_{n_samples}_n-prb_{n_prb}_n-tx_{n_tx}_n-rx_{n_rx}.npy'

def find_class_outliers(embeddings: np.ndarray, labels: np.ndarray, std_threshold: float = 10) -> dict:
    """
    Find outliers for each class that are more than std_threshold standard deviations from the mean.
    
    Args:
        embeddings: UMAP embeddings of shape (n_samples, 2)
        labels: Array of labels for each point
        std_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Dictionary mapping class indices to arrays of outlier indices
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

def get_mask_no_outliers(embedding: np.ndarray, outliers_dict: dict) -> np.ndarray:
    """
    Get a mask to remove outliers from the data.
    
    Args:
        embedding: UMAP embeddings
        outliers_dict: Dictionary of outliers by class
        
    Returns:
        Boolean mask indicating which points to keep
    """
    keep_mask = np.ones(len(embedding), dtype=bool)
    for _, outlier_indices in outliers_dict.items():
        curr_mask = ~np.isin(np.arange(len(embedding)), outlier_indices)
        keep_mask &= curr_mask
    return keep_mask

def print_outliers(outliers: dict, models: list) -> None:
    """
    Print information about outliers found in each class.
    
    Args:
        outliers: Dictionary of outliers by class
        models: List of model names
    """
    print("\nOutlier indices by class:")
    for class_idx, outlier_indices in outliers.items():
        print(f"Class {class_idx} ({models[class_idx]}): {len(outlier_indices)} outliers")
        print(f"  Indices: {outlier_indices}") 