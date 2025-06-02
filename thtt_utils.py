"""Utility functions for training and testing channel models.

This module provides helper functions for data processing, model training,
and testing in the channel model comparison project.
"""

import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from csinet_train_test import test_from_csv, train_model_loop


def nmse(A: np.ndarray, B: np.ndarray) -> float:
    """Calculate Normalized Mean Square Error between two matrices."""
    return (np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro'))**2

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
                        train_csv: str = None,
                        val_csv: str = None,
                        test_csv: str = None) -> None:
    """
    Split data into train/val/test sets and save indices to CSV files.
    
    Args:
        n_samples: Total number of samples
        train_val_test_split: List of proportions for train/val/test split
        seed: Random seed for reproducibility
        train_csv: Path to save training indices
        val_csv: Path to save validation indices
        test_csv: Path to save test indices
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_val_test_split[0])
    val_size = int(n_samples * train_val_test_split[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    if train_csv:
        np.savetxt(train_csv, train_indices, fmt='%d', delimiter=',')
    if val_csv:
        np.savetxt(val_csv, val_indices, fmt='%d', delimiter=',')
    if test_csv:
        np.savetxt(test_csv, test_indices, fmt='%d', delimiter=',')

def train_models(models: list, 
                data_matrices: dict,
                dataset_main_folder: str,
                models_folder: Optional[str] = None,
                encoded_dim: int = 128,
                NC: int = 16,
                train_batch_size: int = 16,
                test_batch_size: int = 1024,
                num_epochs: int = 120,
                learning_rate: float = 1e-2,
                n_refine_nets: int = 5,
                n_runs: int = 1,
                n_train_samples: int = 10240*20,
                n_val_samples: int = 10000,
                n_test_samples: int = 10000,
                seed: int = 2) -> Tuple[list, List[str]]:
    """
    Train models for each area.
    
    Args:
        models: List of model names
        data_matrices: Dictionary of data matrices
        dataset_main_folder: Base folder for datasets
        encoded_dim: Dimension of encoded representation
        NC: Number of delay taps
        train_batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        n_refine_nets: Number of refinement networks
        n_runs: Number of training runs
        n_train_samples: Number of training samples
        n_val_samples: Number of validation samples
        n_test_samples: Number of test samples
        seed: Random seed for reproducibility
        
    Returns:
        List of training results for each model
        List of trained model paths
    """
    models_folder = dataset_main_folder if not models_folder else models_folder
    os.makedirs(models_folder, exist_ok=True)

    all_res = []
    all_model_names = []
    
    for model_idx, model in enumerate(models):
        print(f'Training in area {model_idx} ({model})')
        
        dataset_folder = os.path.join(dataset_main_folder, f'model_{model}')
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Convert channel to angle-delay domain
        ch = convert_channel_angle_delay(data_matrices[model])[:,:,:,:NC]
        
        # Save channel data
        scipy.io.savemat(os.path.join(dataset_folder, 'channel_ad_clip.mat'), 
                        {'all_channel_ad_clip': np.swapaxes(ch, -1, -2)})
        
        # Split data into train/val/test
        train_val_test_split(ch.shape[0], 
                           train_val_test_split=[0.8, 0.1, 0.1], 
                           seed=seed,
                           train_csv=os.path.join(dataset_folder, 'train_data_idx.csv'),
                           val_csv=os.path.join(dataset_folder, 'val_data_idx.csv'),
                           test_csv=os.path.join(dataset_folder, 'test_data_idx.csv'))
        
        model_name = os.path.join(models_folder, f'model_encoded-dim={encoded_dim}_model={model}.path')

        # Train model
        ret = train_model_loop(
            training_data_folder=dataset_folder,
            testing_data_folder=dataset_folder,
            train_csv="train_data_idx.csv",
            val_csv="val_data_idx.csv",
            test_csv="test_data_idx.csv",
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            n_runs=n_runs,
            list_number_training_datapoints=[n_train_samples],
            n_val_samples=n_val_samples,
            n_test_samples=n_test_samples,
            Nc=ch.shape[-1],
            Nt=ch.shape[2],
            encoded_dim=encoded_dim,
            num_epoch=num_epochs,
            tensorboard_writer=True,
            model_path_save=model_name,
            save_model=True,
            lr=learning_rate,
            n_refine_nets=n_refine_nets
        )
        all_res.append(ret)
        all_model_names.append(model_name)
        
    return all_res, all_model_names

def cross_test_models(models: list,
                     data_matrices: dict,
                     dataset_main_folder: str,
                     models_folder: Optional[str] = None,
                     encoded_dim: int = 128,
                     NC: int = 16,
                     Nt: int = 32,
                     seed: int = 2,
                     skip_same: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """
    Test models across different datasets.
    
    Args:
        models: List of model names
        data_matrices: Dictionary of data matrices
        dataset_main_folder: Base folder for datasets
        models_folder: Folder where to find the models
        encoded_dim: Dimension of encoded representation
        NC: Number of delay taps
        Nt: Number of antennas
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of test results, results matrix)
    """
    models_folder = dataset_main_folder if not models_folder else models_folder
    all_test_results = []
    n_models = len(models)
    results_matrix = np.zeros((n_models, n_models))
    
    for model_idx, model in enumerate(models):  # target dataset
        print(f'Testing in dataset {model}')

        tgt_dataset_folder = os.path.join(dataset_main_folder, f'model_{model}')

        # Create a test set with all data
        n_samp = data_matrices[model].shape[0]
        train_val_test_split(n_samp, 
                           train_val_test_split=[0,0,1], 
                           seed=seed, 
                           test_csv=os.path.join(tgt_dataset_folder, 'all.csv'))
        
        for model_idx2, model2 in enumerate(models):  # source dataset
            if skip_same and model2 == model:
                continue
                
            src_dataset_folder = os.path.join(models_folder, f'model_{model2}')
            src_model_path = os.path.join(src_dataset_folder, 
                                          f'model_encoded-dim={encoded_dim}_model={model2}.path')
            
            # Use appropriate test set
            csv_name = 'test_data_idx.csv' if model_idx == model_idx2 else 'all.csv'

            # Test model
            test_results = test_from_csv(
                csv_folder=tgt_dataset_folder,
                csv_name=csv_name,
                model_path=src_model_path,
                encoded_dim=encoded_dim,
                Nc=NC,
                Nt=Nt
            )

            mean_nmse = np.mean(test_results['test_nmse_all'])
            print(f'Mean NMSE for {model2} -> {model}: {mean_nmse:.3f} = {10*np.log10(mean_nmse):.1f} dB')
            
            # Store results in matrix
            results_matrix[model_idx2, model_idx] = mean_nmse
            all_test_results.append(test_results)
            
    return all_test_results, results_matrix

def plot_training_results(all_res: list, models: list, save_path: str = None) -> None:
    """
    Plot training and validation NMSE for each area.
    
    Args:
        all_res: List of training results
        models: List of model names
        save_path: Optional path to save the plot
    """
    def to_db(x):
        return 10 * np.log10(x)

    plt.figure(figsize=(10, 6), dpi=200)

    # Define colors for each model
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for model_idx, model in enumerate(models):
        result = all_res[model_idx]
        train_nmse_db = to_db(np.array(result['all_train_nmse']))
        val_nmse_db = to_db(np.array(result['all_val_nmse']))
        epochs = np.arange(1, len(train_nmse_db) + 1)
        
        # Use same color for both train and val lines
        plt.plot(epochs, train_nmse_db, '-', label=f'{model}', 
                 linewidth=2, color=colors[model_idx])
        plt.plot(epochs, val_nmse_db, '--', linewidth=2, color=colors[model_idx])
        
        test_nmse_db = to_db(result['test_nmse'])
        print(f'{model} - Final Test NMSE: {test_nmse_db:.2f} dB')

    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('NMSE (dB)', fontsize=16)
    plt.title('Training and Validation NMSE per Area', fontsize=14)
    plt.legend(fontsize=14)#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_test_matrix(results_matrix: np.ndarray, models: list, save_path: str = None) -> None:
    """
    Plot the test results matrix as a heatmap.
    
    Args:
        results_matrix: Matrix of test results
        models: List of model names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8), dpi=200)
    
    results_matrix_db = 10 * np.log10(results_matrix)
    
    # Create heatmap
    plt.imshow(results_matrix_db, cmap='viridis_r')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('NMSE (dB)', fontsize=14)
    
    # Add text annotations
    # (first position of results is the training model, second is the testing model)
    for i in range(len(models)):
        for j in range(len(models)):
            plt.text(j, i, f'{results_matrix_db[i, j]:.1f}',
                    ha='center', va='center', color='white' 
                    if results_matrix_db[i, j] > np.mean(results_matrix_db) else 'black',
                    fontsize=16)
    
    plt.xticks(np.arange(len(models)), models, rotation=45, fontsize=14)
    plt.yticks(np.arange(len(models)), models, fontsize=14)
    plt.title('Cross-Test Results (NMSE in dB)', fontsize=14)
    plt.tight_layout()
    plt.ylabel('Training Model', fontsize=16)
    plt.xlabel('Testing Model', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def train_with_percentages(model_name: str, data_matrices: dict, dataset_main_folder: str,
                           percentages: list, load_model: bool = False, model_path: str = None, 
                           epochs: int = 60, models_folder: Optional[str] = None) -> tuple:
    """Train model with different percentages of data.
    
    Args:
        model_name: Name of the model to train
        data_matrices: Dictionary of data matrices
        dataset_main_folder: Base folder for datasets
        percentages: List of percentages to use for training
        load_model: Whether to load pre-trained model
        model_path: Path to pre-trained UMa model (if load_model is True)
        epochs: Number of training epochs
        
    Returns:
        Tuple of (test_nmse_list, model_name)
    """
    models_folder = dataset_main_folder if not models_folder else models_folder

    # Get data for the model
    model_data = data_matrices[model_name]
    n_samp = model_data.shape[0]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random indices for the full dataset
    all_indices = np.random.permutation(n_samp)
    
    # Create nested sets of indices
    nested_indices = {}
    for p in percentages:
        n_samples = int(n_samp * p)
        nested_indices[p] = all_indices[:n_samples]
    
    # Print the sizes to verify
    for p, indices in nested_indices.items():
        print(f"{p*100}% of data: {len(indices)} samples")
    
    # # Split remaining data for validation and testing
    # val_idxs = all_indices[int(n_samp*.8):int(n_samp*.9)]
    # test_idxs = all_indices[int(n_samp*.9):]
    
    # # Generate CSV indices for each portion
    # dataset_folder = os.path.join(dataset_main_folder, f'model_{model_name}')
    # os.makedirs(dataset_folder, exist_ok=True)
    
    # for p in percentages:
    #     df1 = pd.DataFrame(nested_indices[p], columns=["data_idx"])
    #     df1.to_csv(os.path.join(dataset_folder, f'train_data_idx_v2_{p*100}.csv'), index=False)
    
    # # Save val and test indices
    # df2 = pd.DataFrame(val_idxs, columns=["data_idx"])
    # df3 = pd.DataFrame(test_idxs, columns=["data_idx"])
    # df2.to_csv(os.path.join(dataset_folder, 'val_data_idx_v2.csv'), index=False)
    # df3.to_csv(os.path.join(dataset_folder, 'test_data_idx_v2.csv'), index=False)
    

    # Train with different percentages
    test_nmse_list = []
    
    for p in tqdm(percentages, desc=f"Training {model_name} with different percentages"):
        # Create a subset of the data matrix for this percentage
        subset_data = {model_name: model_data[nested_indices[p]]}
        
        # Train model
        res, _ = train_models(
            [model_name], 
            subset_data,
            dataset_main_folder,
            models_folder=models_folder,
            num_epochs=epochs,
            n_train_samples=len(nested_indices[p]),
            n_val_samples=int(len(nested_indices[p]) * 0.1),  # 10% for validation
            n_test_samples=int(len(nested_indices[p]) * 0.1)  # 10% for testing
        )
        
        test_nmse_list.append(10 * np.log10(res[0]['test_nmse']))
        print(f'Data [{p*100}%]: {model_name} {test_nmse_list[-1]:.1f}dB')
    
    return test_nmse_list, model_name