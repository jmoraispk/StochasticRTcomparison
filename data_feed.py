"""
Data Feed

This module provides a DataFeed class for loading and processing channel data 
from a specified root directory and CSV file. It includes functionality for 
creating samples from the data and loading them into a PyTorch Dataset.
"""

import os
import numpy as np
import torch
import sklearn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from typing import Tuple

# If True, data will be passed directly to data loaders without loading from files
USE_DIRECT_DATA = False

def create_samples(data_root: str = None, 
                  csv_path: str = None, 
                  random_state: int = 0, 
                  num_data_point: int = None, 
                  portion: float = 1.0, 
                  select_data_idx: np.ndarray = None,
                  direct_data: np.ndarray = None,
                  direct_indices: np.ndarray = None,
                  use_direct_data: bool = USE_DIRECT_DATA) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare channel data from either file or direct input.
    
    Args:
        data_root: Base directory containing data files
        csv_path: Path to CSV file with indices
        random_state: Random seed for shuffling
        num_data_point: Number of data points to use
        portion: Portion of data to use
        select_data_idx: Specific indices to select
        direct_data: Direct channel data array
        direct_indices: Direct indices array
        use_direct_data: Whether to use direct data (True) or load from files (False)
        
    Returns:
        Tuple of (channel_data, indices)
    """
    # Load data from appropriate source
    if use_direct_data and direct_data is not None:
        channel_ad_clip = direct_data
        data_idx = direct_indices if direct_indices is not None else np.arange(len(direct_data))
    else:
        # Load from files
        channel_ad_clip = loadmat(os.path.join(data_root, 'channel_ad_clip.mat'))['all_channel_ad_clip']
        if select_data_idx is None:
            data_idx = np.loadtxt(os.path.join(data_root, csv_path), dtype=int)
        else:
            data_idx = select_data_idx
            
        # Remove the idxs where it's 0:
        idxs_to_del = np.unique(np.where(np.abs(channel_ad_clip[:, 0, 0] == 0))[0])
        if len(idxs_to_del) > 0:
            print(f'Dropping 0-channel indices {idxs_to_del}')
        data_idx_filtered = np.array([idx for idx in data_idx if idx not in idxs_to_del])
        data_idx = data_idx_filtered.astype(int)
        
        # Select data using indices
        channel_ad_clip = channel_ad_clip[data_idx, ...]
    
    # Common processing for both data sources
    
    # Shuffle if requested
    if random_state is not None:
        channel_ad_clip, data_idx = sklearn.utils.shuffle(channel_ad_clip, data_idx, random_state=random_state)
    channel_ad_clip = np.squeeze(channel_ad_clip)

    # Handle data point selection
    if select_data_idx is None:  # if no particular data is selected
        if num_data_point:
            channel_ad_clip = channel_ad_clip[:num_data_point, ...]
            data_idx = data_idx[:num_data_point, ...]
        else:
            num_data = data_idx.shape[0]
            p = int(num_data * portion)
            channel_ad_clip = channel_ad_clip[:p, ...]
            data_idx = data_idx[:p, ...]

    # Apply normalization consistently for both data sources
    channel_ad_clip /= np.linalg.norm(channel_ad_clip, ord='fro', axis=(-1,-2), keepdims=True)
    channel_ad_clip = channel_ad_clip / np.expand_dims(channel_ad_clip[:, 0, 0] / 
                                                      np.abs(channel_ad_clip[:, 0, 0]), (1,2))

    return channel_ad_clip, data_idx


class DataFeed(Dataset):
    def __init__(self, data_root=None, csv_path=None, random_state=0, num_data_point=None, 
                 portion=1.0, select_data_idx=None, direct_data=None, direct_indices=None):
        """Initialize DataFeed with either file paths or direct data.
        
        Args:
            data_root: Base directory containing data files
            csv_path: Path to CSV file with indices
            random_state: Random seed for shuffling
            num_data_point: Number of data points to use
            portion: Portion of data to use
            select_data_idx: Specific indices to select
            direct_data: Direct channel data array
            direct_indices: Direct indices array
        """
        # Get data from create_samples with all parameters
        self.channel_ad_clip, self.data_idx = create_samples(
            data_root=data_root,
            csv_path=csv_path,
            random_state=random_state,
            num_data_point=num_data_point,
            portion=portion,
            select_data_idx=select_data_idx,
            direct_data=direct_data,
            direct_indices=direct_indices,
            use_direct_data=USE_DIRECT_DATA
        )

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx, ...]
        channel_ad_clip = self.channel_ad_clip[idx, ...]  # Shape: (1, Nc, Nt)

        data_idx = torch.tensor(data_idx, requires_grad=False)
        channel_ad_clip = torch.tensor(channel_ad_clip, requires_grad=False)
        
        # Convert to real/imag format expected by CsinetPlus
        channel_ad_clip = torch.view_as_real(channel_ad_clip)  # Shape: (1, Nc, Nt, 2)
        
        # Move real/imag dimension to front and remove singleton dimension
        # From (1, Nc, Nt, 2) to (2, Nc, Nt)
        channel_ad_clip = channel_ad_clip.squeeze(0).permute(2, 0, 1)  # Shape: (2, Nc, Nt)
        
        return channel_ad_clip.float(), data_idx.long()


if __name__ == "__main__":
    data_root = data_path = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_1"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    batch_size = 64

    train_loader = DataLoader(DataFeed(data_root, train_csv, portion=1.), batch_size=batch_size)
    channel_ad_clip, data_idx = next(iter(train_loader))

    print('done')


  