"""
Data Feed

This module provides a DataFeed class for loading and processing complex-valued 
channel data into PyTorch tensors with separated real and imaginary parts.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class DataFeed(Dataset):
    """Dataset class for loading complex channel data.
    
    Converts complex-valued channel matrices into PyTorch tensors with
    separated real and imaginary components as channels.
    
    Note: Currently assumes input shape (n_samples, 1, Nt, Nc) but will be
    generalized in the future to handle (n_samples, K, Nt, Nc) where K could
    be number of paths, users, or other dimensions that should be stacked
    with Nt or Nc for processing.
    """
    
    @classmethod
    def from_array(cls, channel_data: np.ndarray) -> 'DataFeed':
        """Create a DataFeed from a complex channel array.
        
        Args:
            channel_data: Complex channel data array of shape (n_samples, K, Nt, Nc)
                Currently K must be 1, but this will be generalized in the future
                to support multiple paths/users/etc.
            
        Returns:
            DataFeed instance that will yield tensors of shape (1, 2, Nc, Nt)
            where the first dimension is batch size 1, second dimension contains
            real and imaginary parts, and Nc/Nt are preserved in their original order.
        """
        if not np.iscomplexobj(channel_data):
            raise ValueError("Input data must be complex-valued")
            
        if len(channel_data.shape) != 4:
            raise ValueError(f"Expected 4D input array (n_samples, K, Nt, Nc), got shape {channel_data.shape}")
            
        # TODO: Remove this constraint when implementing support for K > 1
        # K dimension will need to be stacked with either Nt or Nc depending on use case
        if channel_data.shape[1] != 1:
            raise ValueError(f"Currently K must be 1, got shape {channel_data.shape}")
            
        # Print input stats
        # print(f"\nDataFeed input stats:")
        # print(f"Shape: {channel_data.shape}")
        # print(f"Mean abs: {np.mean(np.abs(channel_data)):.3e}")
        # print(f"Std abs: {np.std(np.abs(channel_data)):.3e}")
        # print(f"Max abs: {np.max(np.abs(channel_data)):.3e}")
        # print(f"Min abs: {np.min(np.abs(channel_data)):.3e}")
        
        # Remove zero channels
        zero_mask = np.abs(channel_data[:, 0, 0, 0]) > 0
        if not np.all(zero_mask):
            print(f"Dropping {np.sum(~zero_mask)} zero channels")
            channel_data = channel_data[zero_mask]
            
        # Normalize by Frobenius norm
        frob_norms = np.sqrt(np.sum(np.abs(channel_data)**2, axis=(1,2,3), keepdims=True))
        channel_data = channel_data / frob_norms
        
        # Normalize phase using first element
        ref_phase = channel_data[:, 0, 0, 0] / np.abs(channel_data[:, 0, 0, 0])
        channel_data = channel_data / ref_phase[:, np.newaxis, np.newaxis, np.newaxis]
            
        instance = cls.__new__(cls)
        instance.channel_ad_clip = channel_data
        instance.data_idx = np.arange(len(channel_data))
        return instance

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        # Get complex channel data and remove K dimension (currently always 1)
        # TODO: When K > 1 is supported, this squeeze will be replaced with
        # appropriate reshaping to stack K with either Nt or Nc
        channel_ad_clip = self.channel_ad_clip[self.data_idx[idx]].squeeze(0)
        
        # Print stats before processing
        # print(f"\nProcessing sample {idx}:")
        # print(f"Input shape: {channel_ad_clip.shape}")
        # print(f"Input mean abs: {np.mean(np.abs(channel_ad_clip)):.3e}")
        
        # Split into real and imaginary parts
        channel_real = np.real(channel_ad_clip)
        channel_imag = np.imag(channel_ad_clip)
        
        # Stack as channels [2, Nt, Nc]
        channel_ad_clip = np.stack([channel_real, channel_imag], axis=0)
        
        # Print stats after stacking
        # print(f"After stacking shape: {channel_ad_clip.shape}")
        # print(f"Real mean abs: {np.mean(np.abs(channel_real)):.3e}")
        # print(f"Imag mean abs: {np.mean(np.abs(channel_imag)):.3e}")
        
        # Transpose to get [2, Nc, Nt] as expected by the model
        channel_ad_clip = np.transpose(channel_ad_clip, (0, 2, 1))
        
        # Add batch dimension to make [1, 2, Nc, Nt]
        channel_ad_clip = channel_ad_clip[np.newaxis, ...]
        
        # Convert to torch tensor
        channel_ad_clip = torch.from_numpy(channel_ad_clip).float()
        
        # Print final stats
        # print(f"Output shape: {channel_ad_clip.shape}")
        # print(f"Output mean abs: {torch.mean(torch.abs(channel_ad_clip)):.3e}")
        
        return channel_ad_clip


if __name__ == "__main__":
    data_root = data_path = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_1"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    batch_size = 64

    train_loader = DataLoader(DataFeed(data_root, train_csv, portion=1.), batch_size=batch_size)
    channel_ad_clip, data_idx = next(iter(train_loader))

    print('done')


  