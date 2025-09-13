import numpy as np
import deepmimo as dm

import os
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm

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


def interpolate_dataset_from_seqs(
    dataset: dm.Dataset | dm.MacroDataset,
    sequences: np.ndarray,
    step_meters: float | None = 0.5,
    points_per_segment: int | None = None
) -> dm.Dataset:
    """Create a new Dataset by interpolating along each sequence of indices.

    This function takes sequences of indices into a dataset and creates a new dataset by interpolating
    between consecutive points in each sequence. The interpolation can be done either:
    - Based on physical distance (step_meters): Points are placed every step_meters along each segment
    - Based on fixed count (points_per_segment): A fixed number of evenly-spaced points per segment

    Args:
        dataset: Source dataset containing the data to interpolate
        sequences: Array of shape [n_sequences, sequence_length] containing indices into dataset
        step_meters: Distance between interpolated points. Set to None to use points_per_segment.
        points_per_segment: Number of points per segment. Set to None to use step_meters.

    Returns:
        A new Dataset containing the interpolated data with shape [n_total_points, ...] where
        n_total_points depends on the interpolation parameters and sequence lengths.

    The following fields are interpolated:
        - rx_pos: Receiver positions [n_points, 3]
        - power, phase, delay: Ray parameters [n_points, n_rays] 
        - aoa_az, aod_az, aoa_el, aod_el: Angles [n_points, n_rays]
        - inter: Interaction types [n_points, n_rays] (copied from first point)
        - inter_pos: Interaction positions [n_points, n_rays, n_interactions, 3] (if present)
    """
    # Unwrap MacroDataset if necessary
    dataset = dataset.datasets[0] if isinstance(dataset, dm.MacroDataset) else dataset

    # Ensure ndarray of ints for sequences
    sequences = np.asarray(sequences, dtype=int)
    n_sequences = len(sequences)

    # Define arrays/fields used
    ray_fields = ['rx_pos', 'power', 'phase', 'delay', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el']
    interpolation_fields = ray_fields + (['inter_pos'] if getattr(dataset, 'inter_pos', None) is not None else [])
    replication_fields = ['inter'] if getattr(dataset, 'inter', None) is not None else []

    # Prepare lists for all segments across all sequences
    start_idx_parts: list[np.ndarray] = []
    end_idx_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []

    # Local references for speed
    rx_pos = dataset.rx_pos

    # Build flattened segment lists and interpolation weights
    for seq_idx in tqdm(range(sequences.shape[0]), desc="Interpolating sequences"):
        seq = np.asarray(sequences[seq_idx], dtype=int)
        if seq.size < 2:
            continue
        for k in range(seq.size - 1):
            i1 = int(seq[k])
            i2 = int(seq[k + 1])

            # Determine number of interpolation points for this segment
            if step_meters is not None and points_per_segment is None:
                # Distance-based interpolation
                seg_dist = float(np.linalg.norm(rx_pos[i2] - rx_pos[i1]))
                n_points = max(1, int(np.ceil(seg_dist / float(step_meters))))
            else:
                # Fixed count interpolation
                n_points = 1 if points_per_segment is None else max(1, int(points_per_segment))

            # Gather indices and weights for this segment
            start_idx_parts.append(np.full(n_points, i1, dtype=int))
            end_idx_parts.append(np.full(n_points, i2, dtype=int))
            t_parts.append(np.linspace(0.0, 1.0, n_points, endpoint=False, dtype=np.float32))

    # Concatenate all segments
    if len(t_parts) > 0:
        start_idx = np.concatenate(start_idx_parts, axis=0)
        end_idx = np.concatenate(end_idx_parts, axis=0)
        t_all = np.concatenate(t_parts, axis=0)
    else:
        start_idx = np.empty((0,), dtype=int)
        end_idx = np.empty((0,), dtype=int)
        t_all = np.empty((0,), dtype=np.float32)

    # Helper to interpolate a field in one shot
    def _interpolate_field(field_array: np.ndarray) -> np.ndarray:
        if start_idx.size == 0:
            # No interpolated points
            return field_array[0:0]
        a = field_array[start_idx]
        b = field_array[end_idx]
        # reshape t for broadcasting to match field dims beyond first
        ratio = t_all.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype, copy=False)
        return a * (1.0 - ratio) + b * ratio

    concatenated_data: dict[str, np.ndarray] = {}

    # Interpolate all interpolation fields at once per field
    for field in interpolation_fields:
        base = dataset[field]
        interp_vals = _interpolate_field(base)
        # Append final endpoints of each sequence
        final_idxs = sequences[:, -1] if n_sequences > 0 else np.array([], dtype=int)
        final_vals = base[final_idxs] if final_idxs.size > 0 else base[0:0]
        if interp_vals.shape[0] > 0:
            concatenated_data[field] = np.concatenate([interp_vals, final_vals], axis=0)
        else:
            concatenated_data[field] = final_vals

    # Replicate interaction fields (copy from first point of each segment)
    for field in replication_fields:
        base = dataset[field]
        if start_idx.size > 0:
            replicated = base[start_idx]
        else:
            replicated = base[0:0]
        final_idxs = sequences[:, -1] if n_sequences > 0 else np.array([], dtype=int)
        final_vals = base[final_idxs] if final_idxs.size > 0 else base[0:0]
        if replicated.shape[0] > 0:
            concatenated_data[field] = np.concatenate([replicated, final_vals], axis=0)
        else:
            concatenated_data[field] = final_vals

    # Create new dataset with shared parameters
    new_dataset_params = {}
    for param in ['scene', 'materials', 'load_params', 'rt_params']:
        if hasattr(dataset, param):
            new_dataset_params[param] = getattr(dataset, param)

    new_dataset_params['n_ue'] = int(concatenated_data['rx_pos'].shape[0])
    new_dataset_params['parent_name'] = dataset.get('parent_name', dataset.name)
    new_dataset_params['name'] = f"{dataset.name}_interp"

    new_dataset = dm.Dataset(new_dataset_params)
    new_dataset.tx_pos = dataset.tx_pos

    # Assign all interpolated/replicated arrays
    for field in interpolation_fields + replication_fields:
        if field in concatenated_data:
            new_dataset[field] = concatenated_data[field]

    return new_dataset

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

def build_xy_all(H_norm: np.ndarray, l_in: int, l_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Build full x/y for all samples using given input length and gap.

    Returns:
        x_all: (n_samples, l_in, features)
        y_all: (n_samples, features)
    """
    seq_len = H_norm.shape[1]
    if l_in is None:
        l_in = seq_len - l_gap - 1
    y_idx = l_in + l_gap - 1
    if l_in <= 0 or l_gap < 0 or (y_idx >= seq_len):
        raise ValueError("Invalid l_in/l_gap for given sequence length")
    x_all = H_norm[:, :l_in]
    y_all = H_norm[:, y_idx]
    return x_all, y_all

def compute_nmse_matrix(models_list: list[str], horizon: int, l_in: int,
                        models_folder: str, data_folder: str,
                        num_tx_antennas: int, batch_size: int = 128) -> np.ndarray:
    """Load each trained model and evaluate NMSE on every dataset.

    Args:
        models_list: List of dataset/model names.
        horizon: Prediction gap (l_gap).
        l_in: Input sequence length.
        models_folder: Folder containing trained weight files.
        data_folder: Folder containing normalized datasets 'H_norm_{name}.npy'.
        num_tx_antennas: Number of transmit antennas (features per I/Q half).
        batch_size: Inference batch size to avoid OOM.

    Returns:
        A matrix of NMSE values where [i, j] is performance of src=i tested on tgt=j.
    """
    from nr_channel_predictor_wrapper import construct_model, load_model_weights, predict

    results = np.zeros((len(models_list), len(models_list)), dtype=float)

    for i, src_model in enumerate(models_list):
        weights_path = f"{models_folder}/{src_model}_{horizon}_best.pth"
        print(f"\nLoading model weights: {weights_path}")

        ch_pred_model = construct_model(num_tx_antennas, hidden_size=128, num_layers=3)
        ch_pred_model = load_model_weights(ch_pred_model, weights_path)

        for j, tgt_model in enumerate(models_list):
            print(f"Testing {src_model} on {tgt_model} (horizon={horizon})")
            H_norm_tgt = np.load(f'{data_folder}/H_norm_{tgt_model}.npy')
            x_all, y_all = build_xy_all(H_norm_tgt, l_in=l_in, l_gap=horizon)

            preds = []
            for start in range(0, x_all.shape[0], batch_size):
                end = min(start + batch_size, x_all.shape[0])
                y_pred_b = predict(ch_pred_model, x_all[start:end])
                preds.append(y_pred_b)
            y_pred = np.concatenate(preds, axis=0)

            results[i, j] = nmse(y_pred, y_all)
            print(f"  NMSE: {10*np.log10(results[i, j]):.1f} dB")

    return results