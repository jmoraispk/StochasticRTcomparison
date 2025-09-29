"""Training and testing utilities for channel model comparison.

This module provides functions for training and testing channel models,
including cross-testing between different models and visualization of results.
It supports both stochastic and ray tracing channel models.

This module handles all PyTorch-specific functionality, including:
- Data loading and batching
- Model training and evaluation
- Loss computation and optimization
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
import sys
from einops import rearrange
from typing import Optional, Dict

# Data Loaders
from torch.utils.data import DataLoader
from data_feed import DataFeed

# Models
from CsinetPlus import CsinetPlus
# from transformerAE import TransformerAE


def set_random_state(seed: int = 42):
    """Set random state for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloader(
    direct_data: np.ndarray,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_samples: Optional[int] = None) -> Optional[DataLoader]:
    """Create a PyTorch DataLoader for a specific dataset split.
    
    Args:
        direct_data: Channel data array
        indices: Indices for selecting data
        batch_size: Batch size for the loader
        num_samples: Number of samples to use (None = use all)
        
    Returns:
        DataLoader for the specified data split, or None if indices not provided
    """
    if indices is None:
        return None
        
    return DataLoader(DataFeed(direct_data, indices, num_data_point=num_samples), batch_size=batch_size)

def cal_nmse(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the Normalized Mean Squared Error (NMSE) between complex-valued tensors.

    This function computes NMSE between two complex-valued tensors by:
    1. Rearranging the tensors to proper complex format
    2. Converting to complex tensors
    3. Computing the Frobenius norm-based NMSE

    Args:
        A: Original tensor of shape (batch, RealImag, Nt, Nc)
        B: Approximated tensor of shape (batch, RealImag, Nt, Nc)

    Returns:
        NMSE values for each sample in the batch
    """
    # Calculate the Frobenius norm difference between A and B
    A = rearrange(A, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()
    B = rearrange(B, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()
    A = torch.view_as_complex(A)
    B = torch.view_as_complex(B)
    
    error_norm = torch.norm(A - B, p='fro', dim=(-1, -2))
    A_norm = torch.norm(A, p='fro', dim=(-1, -2))
    
    return (error_norm**2) / (A_norm**2)

def train_model(
    train_loader,
    val_loader,
    test_loader,
    comment="unknown",
    encoded_dim=16,
    num_epoch=200,
    lr=1e-2,
    model_path_save=None,
    model_path_load=None,
    load_model=False,
    save_model=False,
    Nc=100, # Number of subcarriers (delay bins)
    Nt=64,  # Number of antennas    (angle bins)
    n_refine_nets=5,
    seed=42): # number of refine layers at the decoder
    """Train a CSI-Net model on channel data.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        comment: Comment for logging
        encoded_dim: Dimension of encoded representation
        num_epoch: Number of training epochs
        lr: Learning rate
        model_path_save: Path to save model
        model_path_load: Path to load model
        load_model: Whether to load model
        save_model: Whether to save model
        Nc: Number of subcarriers
        Nt: Number of antennas
        n_refine_nets: Number of refinement networks
        
    Returns:
        Dictionary containing training metrics and model path
    """
    # Set random state for reproducibility
    set_random_state(seed)
    
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    
    
    # instantiate the model and send to GPU
    print(f'Creating net with {n_refine_nets} refine nets at decoder side.')
    if n_refine_nets == -1:
        # net = TransformerAE(encoded_dim, Nc, Nt) # kbits=None = no quantization
        pass
    else:
        net = CsinetPlus(encoded_dim, Nc, Nt, n_refine_nets=n_refine_nets)
    net.to(device)

    # path to save the model
    comment = comment + "_" + net.name
    
    if load_model and model_path_load:
        net.load_state_dict(torch.load(model_path_load))
    
    if save_model and not model_path_save:
        model_path_save = "checkpoint/" + comment + ".path"

    # set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 90], gamma=0.5
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # training
    all_train_nmse = []
    all_val_nmse = []
    for epoch in range(num_epoch):
        net.train()
        running_loss = 0.0
        running_nmse = 0.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # get the inputs
                input_channel, data_idx = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                encoded_vector, output_channel = net(input_channel)
                loss = criterion(output_channel, input_channel)

                nmse = torch.mean(cal_nmse(input_channel, output_channel), 0).item()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_nmse = (nmse + i * running_nmse) / (i + 1)
                log = OrderedDict()
                log["loss"] = "{:.6e}".format(running_loss)
                log["nmse"] = running_nmse
                tepoch.set_postfix(log)
            scheduler.step()
        all_train_nmse.append(running_nmse)
        
        if val_loader is None:
            continue  # no validation is needed
        else:
            # validation
            net.eval()
            with torch.no_grad():
                total = 0
                val_loss = 0
                val_nmse = 0

                for data in val_loader:
                    # get the inputs
                    input_channel, data_idx = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    encoded_vector, output_channel = net(input_channel)

                    val_loss += (
                        nn.MSELoss(reduction="mean")(
                            input_channel, output_channel
                        ).item()
                        * data_idx.shape[0]
                    )
                    val_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0)
                    total += data_idx.shape[0]

                val_loss /= float(total)
                val_nmse /= float(total)
            all_val_nmse.append(val_nmse.item())
            print("val_loss={:.3e}".format(val_loss), flush=True)
            print("val_nmse={:.3f}".format(val_nmse), flush=True)
            
    if model_path_save:
        torch.save(net.state_dict(), model_path_save)

    # test
    net.eval()
    with torch.no_grad():
        total = 0
        test_loss = 0
        test_nmse = 0

        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)

            test_loss += (
                nn.MSELoss(reduction="mean")(input_channel, output_channel).item()
                * data_idx.shape[0]
            )
            test_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0).item()
            total += data_idx.shape[0]

        test_loss /= float(total)
        test_nmse /= float(total)

        print("test_loss={:.6e}".format(test_loss), flush=True)
        print("test_nmse={:.6f}".format(test_nmse), flush=True)

        return {
            "all_train_nmse": all_train_nmse,
            "all_val_nmse": all_val_nmse,
            "test_loss": test_loss,
            "test_nmse": test_nmse,
            "model_path": model_path_save,
        }


def test_model(test_loader: DataLoader, 
               net: Optional[nn.Module] = None, 
               model_path: Optional[str] = None, 
               encoded_dim: int = 32, 
               Nc: int = 100, 
               Nt: int = 64, 
               n_refine_nets: int = 5,
               seed: int = 42) -> Dict:
    """Test a trained CSI-Net model on a test dataset.
    
    This function either:
    1. Uses a provided model directly, or
    2. Loads a model from a checkpoint file
    
    It then evaluates the model on the test dataset and returns detailed metrics.
    
    Args:
        test_loader: DataLoader containing test data
        net: Optional pre-initialized model
        model_path: Path to saved model weights
        encoded_dim: Dimension of encoded representation
        Nc: Number of subcarriers
        Nt: Number of antennas
        n_refine_nets: Number of refinement networks
        
    Returns:
        Dictionary containing:
        - test_loss_all: Loss values for each test sample
        - test_nmse_all: NMSE values for each test sample
        - test_data_idx: Indices of test samples
        - inputs: Original input channels
        - encoded: Encoded representations
        - outputs: Reconstructed channels
    """
    # Set random state for reproducibility
    set_random_state(seed)
    
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print('not running on gpu!')

    # instantiate the model and send to GPU
    if model_path:
        net = CsinetPlus(encoded_dim, Nc, Nt, n_refine_nets=n_refine_nets)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    net.to(device)

    # test
    net.eval()
    with torch.no_grad():
        test_loss = []
        test_nmse = []
        test_data_idx = []
        inputs = []
        encoded_vects = []
        outputs = []
        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)
            
            test_loss.append(nn.MSELoss(reduction="none")(input_channel, output_channel).mean((-1,-2,-3)).cpu().numpy())
            test_nmse.append(cal_nmse(input_channel, output_channel).cpu().numpy())
            test_data_idx.append(data_idx.cpu().numpy())            
            encoded_vects.append(encoded_vector.cpu().numpy())
            
            inputs_rearranged = rearrange(input_channel, 'Batch RealImag Nt Nc -> Batch Nt Nc RealImag')
            inputs.append(torch.view_as_complex(inputs_rearranged.contiguous()).cpu().numpy())
            
            outputs_rearranged = rearrange(output_channel, 'Batch RealImag Nt Nc -> Batch Nt Nc RealImag')
            outputs.append(torch.view_as_complex(outputs_rearranged.contiguous()).cpu().numpy())
            
        test_loss = np.concatenate(test_loss)
        test_nmse = np.concatenate(test_nmse)
        test_data_idx = np.concatenate(test_data_idx)
        inputs = np.concatenate(inputs)
        encoded_vects = np.concatenate(encoded_vects)
        outputs = np.concatenate(outputs)
        
        return {
            "test_loss_all": test_loss,
            "test_nmse_all": test_nmse,
            "test_data_idx": test_data_idx,
            "inputs": inputs,
            "encoded": encoded_vects,
            "outputs": outputs,
        }