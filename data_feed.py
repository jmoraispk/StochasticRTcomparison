"""
Data Feed

This module provides a DataFeed class for loading and processing channel data.
It includes functionality for creating samples from direct data input and 
loading them into a PyTorch Dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

def create_samples(direct_data: np.ndarray,
                  indices: np.ndarray = None,
                  num_data_point: int = None, 
                  portion: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare channel data from direct input.
    
    Args:
        direct_data: Direct channel data array (required)
        indices: Specific indices to select from direct_data
        num_data_point: Number of data points to use
        portion: Portion of data to use
        
    Returns:
        Tuple of (channel_data, indices)
    """   
    # Use provided indices or create sequential ones
    data_idx = indices if indices is not None else np.arange(len(direct_data))
    channel_ad_clip = direct_data[data_idx]
    
    channel_ad_clip = np.squeeze(channel_ad_clip)

    # Handle data point selection
    if num_data_point:
        channel_ad_clip = channel_ad_clip[:num_data_point, ...]
        data_idx = data_idx[:num_data_point, ...]
    else:
        num_data = data_idx.shape[0]
        p = int(num_data * portion)
        channel_ad_clip = channel_ad_clip[:p, ...]
        data_idx = data_idx[:p, ...]

    # Apply normalization
    channel_ad_clip /= np.linalg.norm(channel_ad_clip, ord='fro', axis=(-1,-2), keepdims=True)
    channel_ad_clip = channel_ad_clip / np.expand_dims(channel_ad_clip[:, 0, 0] / 
                                                       np.abs(channel_ad_clip[:, 0, 0]), (1,2))

    return channel_ad_clip, data_idx


class DataFeed(Dataset):
    def __init__(self, direct_data: np.ndarray, indices: np.ndarray = None,
                 num_data_point: int = None, portion: float = 1.0):
        """Initialize DataFeed with direct data.
        
        Args:
            direct_data: Direct channel data array (required)
            indices: Specific indices to select from direct_data
            num_data_point: Number of data points to use
            portion: Portion of data to use
        """
        # Get data from create_samples with all parameters
        self.channel_ad_clip, self.data_idx = create_samples(
            direct_data=direct_data,
            indices=indices,
            num_data_point=num_data_point,
            portion=portion
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
    # Example usage with direct data
    data = np.random.randn(100, 1, 32, 64) + 1j * np.random.randn(100, 1, 32, 64)
    train_indices = np.arange(80)  # Use first 80 samples for training
    
    train_loader = DataLoader(
        DataFeed(direct_data=data, indices=train_indices), 
        batch_size=64
    )
    channel_ad_clip, data_idx = next(iter(train_loader))

    print('done')


  