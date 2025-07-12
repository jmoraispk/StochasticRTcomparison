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

from csinet_train_test import train_model, create_dataloader, test_model
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
                        seed: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets.
    
    Args:
        n_samples: Total number of samples
        train_val_test_split: List of proportions for train/val/test split
        seed: Random seed for reproducibility
        
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
    
    for model in models:
        print(f'Training Model {model}')
        
        dataset_folder = config.get_dataset_folder(model)
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Convert channel to angle-delay domain
        ch = convert_channel_angle_delay(data_matrices[model])[:,:,:,:config.n_taps]
        
        # Split data into train/val/test
        train_indices, val_indices, test_indices = train_val_test_split(
            ch.shape[0], 
            train_val_test_split=[0.8, 0.1, 0.1], 
            seed=config.seed
        )

        # Prepare data in same format as file-based mode
        ch_prepared = np.swapaxes(ch, -1, -2)
           
        # Create individual data loaders
        train_loader = create_dataloader(ch_prepared, train_indices,
                                         config.train_batch_size,
                                         config.n_train_samples)
        
        val_loader = create_dataloader(ch_prepared, val_indices,
                                       config.test_batch_size,
                                       config.n_val_samples)
        
        test_loader = create_dataloader(ch_prepared, test_indices,
                                        config.test_batch_size,
                                        config.n_test_samples)

        # Get model paths
        if config.is_finetuning:
            # When fine-tuning, load from source model and save to fine-tuned path
            model_path_load = config.get_model_path(config.source_model)  # Load from source
            model_path_save = config.get_model_path(model)  # Save to fine-tuned path
        else:
            # Normal training
            model_path_save = config.get_model_path(model)
            model_path_load = None

        # Train model with deterministic behavior
        ret = train_model(
            train_loader, val_loader, test_loader,
            comment=f"model_{model}",
            Nc=ch.shape[-1], Nt=ch.shape[2], encoded_dim=config.encoded_dim,
            num_epoch=config.num_epochs,
            model_path_save=model_path_save,
            model_path_load=model_path_load,
            save_model=True,
            load_model=bool(model_path_load),
            lr=config.learning_rate,
            n_refine_nets=config.n_refine_nets,
            seed=config.seed
        )
        
        all_res.append(ret)
        
    return all_res

def cross_test_models(data_matrices: Dict[str, np.ndarray],
                     config: ModelConfig,
                     skip_same: bool = False,
                     use_finetuned: bool = False) -> Tuple[List[Dict], np.ndarray]:
    """Test models across different datasets.
    
    Args:
        data_matrices: Dictionary mapping model names to their data matrices.
                      The keys define which models will be tested.
        config: Model configuration object
        skip_same: Whether to skip testing the same model
        use_finetuned: Whether to use fine-tuned models for cross-testing.
                      If True, loads models from fine-tuned folder.
                      If False, uses base models from original folder.

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
    
    for target_idx, target_model in enumerate(models):  # target dataset
        print(f'Testing on dataset {target_model}')
        
        # Convert target data to angle-delay domain
        ch = convert_channel_angle_delay(data_matrices[target_model])[:,:,:,:config.n_taps]
        ch_prepared = np.swapaxes(ch, -1, -2)
        
        # Create a test set with all data
        _, _, test_indices = train_val_test_split(ch.shape[0], 
            train_val_test_split=[0, 0, 1], seed=config.seed)
        
        # Create test loader with all data for cross-testing
        test_loader = create_dataloader(
            ch_prepared,
            test_indices,
            config.test_batch_size
        )
        
        for source_idx, source_model in enumerate(models):  # source dataset
            if skip_same and source_model == target_model:
                continue
                
            # Determine which model to load based on testing mode
            if use_finetuned and source_model != target_model:
                # Use fine-tuned model for cross-model testing
                test_config = config.clone(
                    is_finetuning=True,
                    source_model=source_model
                )
                model_path = test_config.get_model_path(target_model)
                model_type = "fine-tuned"
            else:
                # Use base model (either same-model testing or when use_finetuned=False)
                test_config = config.clone(is_finetuning=False)
                model_path = test_config.get_model_path(source_model)
                model_type = "original"

            # Test model with deterministic behavior
            test_results = test_model(
                test_loader=test_loader,
                model_path=model_path,
                encoded_dim=config.encoded_dim,
                Nc=config.n_taps,
                Nt=config.n_antennas,
                n_refine_nets=config.n_refine_nets,
                seed=config.seed
            )

            mean_nmse = np.mean(test_results['test_nmse_all'])
            
            # Print progress with clear indication of which model we're using
            if source_model == target_model:
                print(f'Testing original {source_model}: {10*np.log10(mean_nmse):.1f} dB')
            else:
                print(f'Testing {model_type} {source_model} model on {target_model}: {10*np.log10(mean_nmse):.1f} dB')
            
            # Store results in matrix
            results_matrix[source_idx, target_idx] = mean_nmse
            all_test_results.append(test_results)
            
    return all_test_results, results_matrix

