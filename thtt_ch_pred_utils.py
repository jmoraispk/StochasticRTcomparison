import numpy as np
import deepmimo as dm

import os
import matplotlib.pyplot as plt
import subprocess


def db(x):
    return 10 * np.log10(x)


def mse(pred, target):
    """Calculate mean squared error between prediction and target."""
    return np.mean(np.abs(pred - target) ** 2)


def nmse(pred, target):
    """Calculate normalized (by power) MSE between prediction and target."""
    return mse(pred, target) / np.mean(np.abs(target) ** 2)

########### SEQUENCING UTILS - to select points in the grid ###########

def get_consecutive_active_segments(dataset: dm.Dataset, idxs: np.ndarray,
                                    min_len: int = 1) -> list[np.ndarray]:
    """Get consecutive segments of active users.
    
    Args:
        dataset: DeepMIMO dataset
        idxs: Array of user indices to check
        
    Returns:
        List of arrays containing consecutive active user indices
    """
    active_idxs = np.where(dataset.los[idxs] != -1)[0]
    
    # Split active_idxs into arrays of consecutive indices
    splits = np.where(np.diff(active_idxs) != 1)[0] + 1
    consecutive_arrays = np.split(active_idxs, splits)
    
    # Filter out single-element arrays
    consecutive_arrays = [idxs[arr] for arr in consecutive_arrays if len(arr) > min_len]
    
    return consecutive_arrays


def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
    n_cols, n_rows = dataset.grid_size
    all_seqs = []
    for k in range(n_rows):
        idxs = dataset.get_row_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    for k in range(n_cols):
        idxs = dataset.get_col_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    return all_seqs


def make_sequence_video(dataset, folder='sweeps', ffmpeg_fps=60):
    """
    Generate a video visualizing all row/col user sequences in the dataset.

    For each row and column, plots the consecutive active user segments and saves as PNGs.
    Then, uses ffmpeg to combine the PNGs into a video.

    Args:
        dataset: DeepMIMO dataset object.
        folder: Output folder for PNGs and video.
        ffmpeg_fps: Framerate for the output video.
    """

    os.makedirs(folder, exist_ok=True)
    n_cols, n_rows = dataset.grid_size

    for row_or_col in ['row', 'col']:
        for k in range(n_rows if row_or_col == 'row' else n_cols):
            idx_func = dataset.get_row_idxs if row_or_col == 'row' else dataset.get_col_idxs
            idxs = idx_func(k)
            consecutive_arrays = get_consecutive_active_segments(dataset, idxs)
            
            print(f"{row_or_col} {k} has {len(consecutive_arrays)} consecutive segments:")
            dataset.los.plot()
            for i, arr in enumerate(consecutive_arrays):
                print(f"Segment {i}: {len(arr)} users")
                plt.scatter(dataset.rx_pos[arr, 0], 
                            dataset.rx_pos[arr, 1], color='red', s=.5)
            
            plt.savefig(f'{folder}/asu_campus_3p5_{row_or_col}_{k:04d}.png', 
                        bbox_inches='tight', dpi=200)
            plt.close()
            # break

    # Create video from PNGs using ffmpeg
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(ffmpeg_fps),
        "-pattern_type", "glob",
        "-i", f"{folder}/*.png",
        "-vf", "crop=in_w:in_h-mod(in_h\\,2)",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        f"{folder}/output_{ffmpeg_fps}fps.mp4"
    ])

########### Data Generation INTERPOLATION - for ray tracing ###########


# Make a function that interpolates the path between 2 users
def interpolate_percentage(array1, array2, percents):
    """Interpolate between two points at specified percentages.
    
    Args:
        pos1: Starting position/value
        pos2: Ending position/value
        percents: Array of percentages between 0 and 1
        
    Returns:
        np.ndarray: Array of interpolated values at given percents
    """
    # Ensure percentages are between 0 and 1
    percents = np.clip(percents, 0, 1)

    # Broadcast to fit shape of interpolated array
    percents = np.reshape(percents, percents.shape + (1,) * array1.ndim)

    return array1 * (1 - percents) + array2 * percents




def get_all_sequences(dataset: dm.Dataset, min_len: int = 1) -> list[np.ndarray]:
    """
    Extract all consecutive active user index sequences from a dataset, 
    considering both rows and columns of the grid.

    For each row and each column in the dataset grid, this function finds all
    consecutive segments of active users (with length at least `min_len`) and
    returns them as a list of index arrays.

    Args:
        dataset (dm.Dataset): The dataset object, expected to provide grid_size,
            get_row_idxs, get_col_idxs, and to be compatible with
            get_consecutive_active_segments.
        min_len (int, optional): Minimum length of a segment to be included. 
            Defaults to 1.

    Returns:
        list[np.ndarray]: List of arrays, each containing indices of a consecutive
            active user segment (row-wise or column-wise).
    """
    n_cols, n_rows = dataset.grid_size
    all_seqs = []
    for k in range(n_rows):
        idxs = dataset.get_row_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    for k in range(n_cols):
        idxs = dataset.get_col_idxs(k)
        consecutive_arrays = get_consecutive_active_segments(dataset, idxs, min_len)
        all_seqs += consecutive_arrays

    return all_seqs


def expand_to_uniform_sequences(
    sequences: list[np.ndarray] | np.ndarray,
    target_len: int,
    stride: int = 1
) -> np.ndarray:
    """
    Convert a list or array of index sequences into a 2D array of fixed-length windows.

    For each input sequence, this function extracts all possible contiguous subsequences
    (windows) of length `target_len` using a sliding window with the specified `stride`.
    Sequences shorter than `target_len` are ignored.

    Args:
        sequences (list[np.ndarray] | np.ndarray): List of 1D arrays or a 2D array,
            where each element/row is a sequence of indices.
        target_len (int): Desired length of each output window.
        stride (int, optional): Step size for the sliding window. Defaults to 1.

    Returns:
        np.ndarray: 2D array of shape (n_windows, target_len), where each row is a
            window of indices from the input sequences. If no windows are found,
            returns an empty array of shape (0, target_len).
    """
    if isinstance(sequences, list):
        seq_list = [np.asarray(seq, dtype=int) for seq in sequences]
    else:
        # sequences is assumed 2D already; convert to list of 1D arrays
        seq_list = [np.asarray(sequences[i], dtype=int) for i in range(sequences.shape[0])]

    out: list[np.ndarray] = []
    for seq in seq_list:
        if len(seq) < target_len:
            continue
        for i in range(0, len(seq) - target_len + 1, stride):
            out.append(seq[i:i+target_len])
    if len(out) == 0:
        return np.empty((0, target_len), dtype=int)
    return np.stack(out, axis=0)



########### Data Generation POST-PROCESSING - Normalize & Save ###########

def process_and_save_channel(H_complex: np.ndarray,
                             time_axis: int,
                             data_folder: str,
                             model: str,
                             snr_db: float,
                             save: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Standardize, real-split, add AWGN, normalize, and save channel data.

    This function unifies the final processing steps for both DeepMIMO (ray tracing)
    and Sionna (stochastic) channel tensors so they follow the same route.

    Steps:
    1) Move the time dimension to axis 1 so arrays become (n_samples, seq_len, ...)
    2) Flatten all remaining feature axes into a single feature axis (n_samples, seq_len, features)
    3) Concatenate real and imaginary parts along the feature axis (n_samples, seq_len, real_features)
    4) Add AWGN with the given SNR in dB
    5) Normalize by the max absolute value of the noisy signal
    6) Save H_norm and H_noisy_norm as float32 to data_folder using the model name

    Args:
        H_complex: Complex channel array. Batch axis must be 0. Time axis can be arbitrary.
        time_axis: The axis index of time in H_complex (e.g., 1 for DeepMIMO sequences,
                   -1 for Sionna format).
        data_folder: Target folder to save outputs.
        model: String identifier used in file names.
        snr_db: SNR in dB to generate additive white Gaussian noise.

    Returns:
        H_norm: processed array of shape (n_samples, seq_len, real_features)
        H_noisy_norm: processed array of shape (n_samples, seq_len, real_features)
        h_max: normalization factor used
    """
    # Bring time axis to position 1: (n_samples, seq_len, ...)
    H_std = np.moveaxis(H_complex, time_axis, 1)

    # Flatten remaining feature dimensions
    n_samples, seq_len = H_std.shape[0], H_std.shape[1]
    H_flat = H_std.reshape(n_samples, seq_len, -1)

    # Split real/imag into features
    H_realimag = np.concatenate([H_flat.real, H_flat.imag], axis=-1)

    # Add AWGN
    noise_var = 10 ** (-snr_db / 10.0)
    noise = np.random.randn(*H_realimag.shape) * np.sqrt(noise_var)
    H_noisy = H_realimag + noise

    # Normalize by max abs of noisy
    h_max = np.nanmax(np.abs(H_noisy))
    H_noisy_norm = (H_noisy / h_max).astype(np.float32)
    H_norm = (H_realimag / h_max).astype(np.float32)

    # Save
    if save:
        os.makedirs(data_folder, exist_ok=True)
        np.save(f"{data_folder}/H_norm_{model}.npy", H_norm)
        np.save(f"{data_folder}/H_noisy_norm_{model}.npy", H_noisy_norm)

    return H_norm, H_noisy_norm, h_max

########### Model Training PRE-PROCESSING - Split data ###########

def split_data(H_norm: np.ndarray, train_ratio: float = 0.9, 
               l_in: int | None = None, l_gap: int = 0):
    """
    Parameters
    ----------
    H_norm : np.ndarray
        Array shaped (n_samples, seq_len).
    train_ratio : float, default 0.9
        Fraction of samples used for training.
    l_in : int, optional
        Number of consecutive time-steps fed to the model (x).  
        If None, use all available steps except the prediction target and gap.
    l_gap : int, default 0
        Number of steps skipped between the end of x and the prediction target y.
        If l_gap = 0, use the last input step as prediction target.

    Returns
    -------
    x_train, y_train, x_val, y_val : np.ndarray
    """

    seq_len = H_norm.shape[1]

    # default ‑ keep old behaviour ⇒ use every step except the last one
    if l_in is None:
        l_in = seq_len - l_gap - 1

    y_idx = l_in + l_gap - 1

    if l_in <= 0 or l_gap < 0 or (y_idx >= seq_len):
        raise ValueError("Invalid l_in/l_gap for given sequence length")

    # build input and target
    x_all = H_norm[:, :l_in]             # first l_in steps
    y_all = H_norm[:, y_idx]  # one step after the gap

    n_samples = H_norm.shape[0]
    n_train = int(n_samples * train_ratio)

    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val   = x_all[n_train:]
    y_val   = y_all[n_train:]

    # quick sanity print‑outs
    print(f"x_train.shape: {x_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"x_val.shape: {x_val.shape}")
    print(f"y_val.shape: {y_val.shape}")

    return x_train, y_train, x_val, y_val