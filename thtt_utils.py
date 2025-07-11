"""Utility functions for training and testing channel models.

This module provides helper functions for data processing, model training,
and testing in the channel model comparison project. It handles data preparation,
file management, and results visualization, while delegating PyTorch-specific
operations to csinet_train_test.py.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from csinet_train_test import train_model, create_dataloaders, test_model
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
        
        # Convert channel to angle-delay domain
        ch = convert_channel_angle_delay(data_matrices[model])[:,:,:,:config.n_taps]
        
        # Create data loaders directly from channel data
        train_loader, val_loader, test_loader = create_dataloaders(
            data=ch,
            n_train_samples=config.n_train_samples,
            n_val_samples=config.n_val_samples,
            n_test_samples=config.n_test_samples,
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            random_state=config.seed
        )

        # Get model paths - for fine-tuning, load from source model
        model_path_save = config.get_model_path(model)
        model_path_load = config.get_pretrained_path(model) if config.is_finetuning else None

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
            load_model=config.is_finetuning,
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
        
        # Convert channel to angle-delay domain
        ch = convert_channel_angle_delay(data_matrices[model])[:,:,:,:config.n_taps]
        
        # Create test loader directly from data
        _, _, test_loader = create_dataloaders(
            data=ch,
            n_test_samples=config.n_test_samples,
            test_batch_size=config.test_batch_size,
            random_state=config.seed
        )
        
        for model_idx2, model2 in enumerate(models):  # source dataset
            if skip_same and model2 == model:
                continue
                
            # Get model path based on whether it's a fine-tuned model
            if config.is_finetuning:
                model_path = config.get_model_path(model2)  # Load from fine-tuned folder
            else:
                model_path = os.path.join(config.dataset_main_folder, f"model_{model2}.pth")

            # Test model
            test_results = test_model(
                test_loader=test_loader,
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

