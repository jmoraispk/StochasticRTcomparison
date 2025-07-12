"""Utility functions for training and testing channel models.

This module provides helper functions for data processing, model training,
and testing in the channel model comparison project. It handles data preparation,
file management, and results visualization, while delegating PyTorch-specific
operations to csinet_train_test.py.
"""

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.io

from csinet_train_test import test_from_csv, train_model, create_dataloaders
from model_config import ModelConfig

def convert_channel_angle_delay(channel: np.ndarray) -> np.ndarray:
    """
    Convert channel to angle-delay domain.
    
    Args:
        channel: Input channel matrix
        
    Returns:
        Channel matrix in angle-delay domain
    """
    # Apply FFT along the last dimension (delay domain)
    return np.fft.fft(channel, axis=-1)

def train_val_test_split(n_samples: int,
                        train_val_test_split: List[float] = [0.8, 0.1, 0.1],
                        seed: int = 2,
                        train_csv: Optional[str] = None,
                        val_csv: Optional[str] = None,
                        test_csv: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets and save indices to CSV files.
    
    Args:
        n_samples: Total number of samples
        train_val_test_split: List of proportions for train/val/test split
        seed: Random seed for reproducibility
        train_csv: Path to save training indices
        val_csv: Path to save validation indices
        test_csv: Path to save test indices
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_val_test_split[0])
    val_size = int(n_samples * train_val_test_split[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Always save indices to CSV files if paths are provided
    if train_csv:
        np.savetxt(train_csv, train_indices, fmt='%d', delimiter=',')
    if val_csv:
        np.savetxt(val_csv, val_indices, fmt='%d', delimiter=',')
    if test_csv:
        np.savetxt(test_csv, test_indices, fmt='%d', delimiter=',')
        
    return train_indices, val_indices, test_indices

def train_models(data_matrices: Dict[str, np.ndarray], config: ModelConfig) -> List[Dict]:
    """Train models for each area.
    
    Args:
        data_matrices: Dictionary mapping model names to their data matrices.
                      The keys define which models will be trained.
        config: Model configuration object containing all training parameters,
               including paths to any pretrained models if doing fine-tuning.
        
    Returns:
        List of training results for each model
    """
    models = list(data_matrices.keys())
    all_res = []
    
    for model_idx, model in enumerate(models):
        print(f'Training in area {model_idx} ({model})')
        
        dataset_folder = config.get_dataset_folder(model)
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Convert channel to angle-delay domain
        print(f"Initial data shape: {data_matrices[model].shape}")
        ch = convert_channel_angle_delay(data_matrices[model])[:,:,:,:config.n_taps]
        print(f"After angle-delay conversion: {ch.shape}")
        
        # Split data into train/val/test and always save indices
        train_indices, val_indices, test_indices = train_val_test_split(
            ch.shape[0], 
            train_val_test_split=[0.8, 0.1, 0.1], 
            seed=config.seed,
            train_csv=os.path.join(dataset_folder, 'train_data_idx.csv'),
            val_csv=os.path.join(dataset_folder, 'val_data_idx.csv'),
            test_csv=os.path.join(dataset_folder, 'test_data_idx.csv')
        )

        # Always save channel data to .mat file
        print('Saving channel data to .mat file')
        ch_prepared = np.swapaxes(ch, -1, -2)  # Prepare data in same format as file-based mode
        scipy.io.savemat(os.path.join(dataset_folder, 'channel_ad_clip.mat'), 
                        {'all_channel_ad_clip': ch_prepared})
        
        print(f"Data shape after swapaxes: {ch_prepared.shape}")
            
        # Create data loaders - always pass all parameters
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_folder=dataset_folder,
            direct_data=ch_prepared,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            n_train_samples=config.n_train_samples,
            n_val_samples=config.n_val_samples,
            n_test_samples=config.n_test_samples,
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            random_state=config.seed
        )

        # Get model paths - for fine-tuning, config will return pretrained path
        model_path_save = config.get_model_path(model)
        model_path_load = config.get_model_path(model) if config.is_finetuning else None

        # Train model directly using train_model
        ret = train_model(
            train_loader,
            val_loader,
            test_loader,
            comment=f"model_{model}",
            Nc=ch.shape[-1],
            Nt=ch.shape[2],
            encoded_dim=config.encoded_dim,
            num_epoch=config.num_epochs,
            model_path_save=model_path_save,
            model_path_load=model_path_load,
            save_model=True,
            load_model=bool(model_path_load),
            lr=config.learning_rate,
            n_refine_nets=config.n_refine_nets
        )
        
        all_res.append(ret)
        
    return all_res

def cross_test_models(data_matrices: Dict[str, np.ndarray],
                     config: ModelConfig,
                     skip_same: bool = False) -> Tuple[List[Dict], np.ndarray]:
    """Test models across different datasets.
    
    Args:
        data_matrices: Dictionary mapping model names to their data matrices.
                      The keys define which models will be tested.
        config: Model configuration object
        skip_same: Whether to skip testing the same model

    Returns:
        Tuple of:
        - List of test results for each model pair
        - Results matrix where [i,j] is NMSE for model i tested on dataset j
          (rows=source models, columns=target datasets)
    """
    models = list(data_matrices.keys())
    all_test_results = []
    n_models = len(models)
    results_matrix = np.zeros((n_models, n_models))
    
    for model_idx, model in enumerate(models):  # target dataset
        print(f'Testing in dataset {model}')

        tgt_dataset_folder = config.get_dataset_folder(model)

        # Create a test set with all data
        n_samp = data_matrices[model].shape[0]
        train_val_test_split(n_samp, 
                           train_val_test_split=[0,0,1], 
                           seed=config.seed, 
                           test_csv=os.path.join(tgt_dataset_folder, 'all.csv'))
        
        for model_idx2, model2 in enumerate(models):  # source dataset
            if skip_same and model2 == model:
                continue
                
            model_path = config.get_model_path(model2)
            
            # Use appropriate test set
            csv_name = 'test_data_idx.csv' if model_idx == model_idx2 else 'all.csv'

            # Test model
            test_results = test_from_csv(
                csv_folder=tgt_dataset_folder,
                csv_name=csv_name,
                model_path=model_path,
                encoded_dim=config.encoded_dim,
                Nc=config.n_taps,
                Nt=config.n_antennas,
                n_refine_nets=config.n_refine_nets
            )

            mean_nmse = np.mean(test_results['test_nmse_all'])
            print(f'Mean NMSE for {model2} -> {model}: {mean_nmse:.3f} = {10*np.log10(mean_nmse):.1f} dB')
            
            # Store results in matrix
            results_matrix[model_idx2, model_idx] = mean_nmse
            all_test_results.append(test_results)
            
    return all_test_results, results_matrix

