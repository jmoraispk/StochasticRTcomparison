#%% [MANDATORY CONSTANTS - ANY ENV] Imports

import numpy as np
import os
import matplotlib.pyplot as plt

import deepmimo as dm

# To plot H for specific antennas (uses only matplotlib)
from thtt_ch_pred_plot import plot_iq_from_H

NT = 2
NR = 1

SNR = 150 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
MAX_DOOPLER = 40 # [Hz]

def db(x):
    return 10 * np.log10(x)

def mse(pred, target):
    """Calculate mean squared error between prediction and target."""
    return np.mean(np.abs(pred - target) ** 2)

def nmse(pred, target):
    """Calculate normalized (by power) MSE between prediction and target."""
    return mse(pred, target) / np.mean(np.abs(target) ** 2)

#%% [ANY ENV] Load data

# matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
#             'delay', 'power', 'phase', 'inter']
essential_matrices = ['rx_pos', 'aoa_az', 'inter']

# dataset = dm.load('asu_campus_3p5_10cm', matrices=essential_matrices)
dataset = dm.load('asu_campus_3p5')#, matrices=essential_matrices)

#%% [ANY ENV] Generate all linear sequences in a scenario

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
    
#%% [ANY ENV] Make video of all sequences

folder = 'sweeps'
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

import subprocess

subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "60",
    "-pattern_type", "glob",
    "-i", f"{folder}/*.png",
    "-vf", "crop=in_w:in_h-mod(in_h\\,2)",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    f"{folder}/output_60fps.mp4"
])

#%% [ANY ENV] Create all sequences

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

all_seqs = get_all_sequences(dataset, min_len=1)

# Print statistics
seq_lens = [len(seq) for seq in all_seqs]
sum_len_seqs = sum(seq_lens)
avg_len_seqs = sum_len_seqs / len(all_seqs)

print(f"Number of sequences: {len(all_seqs)}")
print(f"Average length of sequences: {avg_len_seqs:.1f}")

print(f"Number of active users: {len(dataset.get_active_idxs())}")
print(f"Total length of sequences: {sum_len_seqs}")

plt.hist(seq_lens, bins=np.arange(1, max(seq_lens) + 1))
plt.xlabel('Sequence length')
plt.ylabel('Number of sequences')
plt.title('Distribution of sequence lengths')
plt.grid()
plt.show()

#%% [ANY ENV] Creating ray tracing data for Channel Prediction

# Split all sequences in _ LENGTH
L = 95
all_trimmed_seqs = []
for seq in all_seqs:
    for i in range(len(seq) - L + 1): # ignores sequences shorter than L
        all_trimmed_seqs.append(seq[i:i+L])

all_seqs_mat_t = np.array(all_trimmed_seqs)
print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

# sample N sequences from all_trimmed_seqs_mat
N = min(100_000, len(all_seqs_mat_t))
idxs = np.random.choice(len(all_seqs_mat_t), N, replace=False)
all_seqs_mat_t2 = all_seqs_mat_t[idxs]
print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
dataset.set_doppler(MAX_DOOPLER)
H = dataset.compute_channels(ch_params)
print(f"H.shape: {H.shape}")

# Take sequences right away (even if some data may repeat - may happen for short sequences)



# Concatenate feature dimensions (rx ant, tx ant, subcarriers)
H2 = H.reshape(H.shape[0], -1)
print(f"H2.shape: {H2.shape}") # (batch_size, features)

# Make Real by putting IQ into the features
H2 = np.concatenate([H2.real, H2.imag], axis=-1)
print(f"H2.shape: {H2.shape}") # (batch_size, 2*features)

# Add noise
noise_var = 10**(-SNR/10)
noise = np.random.randn(*H2.shape) * np.sqrt(noise_var)
H_noisy = H2 + noise

# Normalize (min-max)
h_max = np.nanmax(np.abs(H_noisy))  # Note: for SNR < 100, max h_norm != max(h_noisy)
H_noisy_norm = H_noisy / h_max
H_norm = H2 / h_max
print(f"H_norm.shape: {H_norm.shape}") # (batch_size, features)
print(f"H_noisy_norm.shape: {H_noisy_norm.shape}") # (batch_size, features)

# Apply sequences in all_seqs_mat_t (to generate time dimensions)
H_norm_seq = H_norm[all_seqs_mat_t2, :].astype(np.float32)
H_noisy_norm_seq = H_noisy_norm[all_seqs_mat_t2, :].astype(np.float32)
print(f"H_norm_seq.shape: {H_norm_seq.shape}") # (batch_size, seq_len, features)

# Save data
model = 'asu_campus_3p5'
folder = 'ch_pred_data'
os.makedirs(folder, exist_ok=True)
np.save(f'{folder}/H_norm_{model}.npy', H_norm_seq)
np.save(f'{folder}/H_noisy_norm_{model}.npy', H_noisy_norm_seq)

#%% [SIONNA ENV] Import and Create Data Generator

from tqdm import tqdm
from data_gen import DataConfig
from sionna_ch_gen import SionnaChannelGenerator

# Configure data generation
data_cfg = DataConfig(
    n_samples = 100_000,
    n_prbs = 20,
    n_rx = NR,
    n_tx = NT, 
    n_time_steps = 55 + 40, # 55 for input, 100 for output
    samp_freq = 1e3,
    batch_size = 100,
    seed = 42
)

model = 'TDL-A'
config = data_cfg

fc = 3.5e9 # [Hz]
speed = MAX_DOOPLER / (fc / 3e8) # E.g. at 3 m/s & 3.5 GHz, max Doppler = 37 Hz
 
print(f"Generating stochastic data for {model}...")
ch_gen = SionnaChannelGenerator(num_prbs=config.n_prbs,
                                channel_name=model,
                                batch_size=config.batch_size,
                                n_rx=config.n_rx,
                                n_tx=config.n_tx,
                                normalize=False,
                                seed=config.seed,
                                ue_speed=speed, # [m/s]
                                delay_spread=300e-9, # [s]
                                frequency=fc, # [Hz]
                                subcarrier_spacing=15e3) # [Hz]



#%% [SIONNA ENV] Generate channel data

def channel_sample(batch_size=1000, num_time_steps=10, sampling_frequency=1e3):
    """Sample channel coefficients and delays.
    
    Args:
        batch_size (int): Number of samples to generate
        num_time_steps (int): Number of time steps to generate
        sampling_frequency (float): Sampling frequency in Hz
        
    Returns:
        H (np.ndarray): Channel matrix of shape [batch_size, num_rx_ant, num_tx_ant, num_time_steps]
    """
    if model in ['UMa', 'UMi']:
        a, t = ch_gen.channel(num_time_steps, sampling_frequency)
    else:
        a, t = ch_gen.channel(batch_size=batch_size, 
                              num_time_steps=num_time_steps, 
                              sampling_frequency=sampling_frequency)

    a, t = a.numpy(), t.numpy()
    # a [batch size, num_rx = 1, num_rx_ant, num_tx=1, num_tx_ant,  num_paths, num_time_steps], tf.complex
    # t [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float) [s]

    # Squeeze num_rx and num_tx dimensions in a & t
    a = a[:, 0, :, 0, :, :, :]  # [batch size, num_rx_ant num_tx_ant, num_paths, num_time_steps]
    t = t[:, 0, 0, :]  # [batch size, num_paths]

    # # Calculate phase shifts and sum along paths
    # phase_shifts = np.exp(-1j * 2 * np.pi * ch_gen.fc * t)
    # H = np.sum(a * phase_shifts[:, None, None, :, None], axis=3)
    
    H = a.sum(axis=3)

    # Calculate frequency-domain response at a chosen subcarrier (non-DC)
    # fft_size = data_cfg.n_prbs * 12
    # delta_f = ch_gen.subcarrier_spacing  # [Hz]
    # k_offset = 1  # first subcarrier offset from DC; adjust if needed
    # f_k = k_offset * delta_f

    # phase_shifts = np.exp(-1j * 2 * np.pi * f_k * t)  # [batch size, num_paths]
    # H = np.sum(a * phase_shifts[:, None, None, :, None], axis=3)

    return H  # [batch size, num_rx_ant, num_tx_ant, num_time_steps]

# Generate channel data
H = np.zeros((config.n_samples, data_cfg.n_rx, data_cfg.n_tx, data_cfg.n_time_steps), dtype=np.complex64)

b = config.batch_size
pbar = tqdm(range(config.n_samples // b), desc="Generating channel data")
for i in pbar:
    H[i*b:(i+1)*b] = channel_sample(b, config.n_time_steps, config.samp_freq)

print(f"H.shape: {H.shape}")

# Plot H for specific antennas 
plot_sample_idx, plot_rx_idx = plot_iq_from_H(H)

# Merge antenna dimensions
H2 = H.reshape(H.shape[0], -1, H.shape[-1])
print(f"H2.shape: {H2.shape}") # (batch, n_rx_ant * n_tx_ant, seq_len)

# Put features in the last dimension
H2 = H2.swapaxes(1, 2)
print(f"H2.shape: {H2.shape}") # (batch_size, sequence_length, features)

# Make Real by putting IQ into the features
H2 = np.concatenate([H2.real, H2.imag], axis=2)
print(f"H2.shape: {H2.shape}") # (batch_size, sequence_length, 2*features)

# Add noise
noise_var = 10**(-SNR/10)
noise = np.random.randn(*H2.shape) * np.sqrt(noise_var)
H_noisy = H2 + noise

# Normalize
h_max = np.nanmax(np.abs(H_noisy))
H_noisy_norm = H_noisy / h_max
H_norm = H2 / h_max
print(f"H_noisy_norm.shape: {H_noisy_norm.shape}")

plot_iq_from_H(H / h_max, plot_sample_idx, plot_rx_idx)

# NOTE: either normalize just range (no shift): x = x / (x_max - x_min)
#       or normalize range and shift: x = (x - x_min) / (x_max - x_min)
#       BUT the latter requires a denormalization when computing NMSE

# Save data
folder = 'ch_pred_data'
os.makedirs(folder, exist_ok=True)
np.save(f'{folder}/H_noisy_norm_{model}.npy', H_noisy_norm)
np.save(f'{folder}/H_norm_{model}.npy', H_norm)

#%% [PYTORCH ENVIRONMENT] Split data

from nr_channel_predictor_wrapper import (
    construct_model, train, predict, info, 
    save_model_weights, load_model_weights
)

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

#%% [PYTORCH ENVIRONMENT] Train models

models_folder = 'ch_pred_models'
os.makedirs(models_folder, exist_ok=True)

# models = ['TDL-A', 'CDL-C', 'UMa', 'asu_campus_3p5']
# models = ['TDL-A', 'CDL-C', 'UMa']
# models = ['asu_campus_3p5']
models = ['TDL-A']

horizons = [1, 3, 5, 10, 20]#, 30, 40]
L = 40  # input sequence length

val_loss_per_horizon_gru = {model: [] for model in models}
val_loss_per_horizon_sh = {model: [] for model in models}

for model in models:
    H_norm = np.load(f'ch_pred_data/H_norm_{model}.npy') # (n_samples, seq_len)

    for horizon in horizons:
        print(f"========== Horizon: {horizon} ==========")
        x_train, y_train, x_val, y_val = split_data(H_norm, l_in=L, l_gap=horizon)

        ch_pred_model = construct_model(NT, hidden_size=128, num_layers=2)
        
        info(ch_pred_model)

        trained_model, tr_loss, val_loss, elapsed_time = \
            train(ch_pred_model, x_train, y_train, x_val, y_val, 
                initial_learning_rate=1e-4, batch_size=64, num_epochs=280, 
                verbose=True, patience=30, patience_factor=1)

        save_model_weights(trained_model, f'{models_folder}/{model}_{horizon}.pth')

        # Plot training and validation loss
        plt.plot(db(tr_loss), label='Training')
        plt.plot(db(val_loss), label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('NMSE Loss (dB)')
        plt.title(f'Training and validation loss for {model} horizon {horizon} ms')
        plt.legend()
        plt.grid()
        plt.show()
        
        # sample & hold baseline (i.e. no prediction)
        sh_loss = nmse(x_val[:, -1], y_val)

        val_loss_per_horizon_gru[model].append(db(min(val_loss)))
        # TODO: make sure the best model is saved, not the last one
        val_loss_per_horizon_sh[model].append(db(sh_loss))

#%%
o = 0
plt.figure(dpi=200)
colors = ['red', 'blue', 'green', 'purple']
for i, model in enumerate(models):
    plt.plot(horizons[o:], val_loss_per_horizon_gru[model][o:], label=model, 
             color=colors[i], marker='o', markersize=3)
    plt.plot(horizons[o:], val_loss_per_horizon_sh[model][o:], label=model + '_SH', 
             color=colors[i], linestyle='--', marker='o', markersize=3)
plt.xlabel('Horizon (ms)')
plt.ylabel('Validation Loss (NMSE in dB)')
plt.legend(ncols=4, bbox_to_anchor=(0.46, 1.0), loc='lower center')
plt.xlim(0, horizons[-1] + 0.5)
plt.grid()
plt.show()

#%%

# NOTES:

# 1. TDL has no antenna correlation (unless we introduce it!)
# 2. CDL has ant. corr. but needs the antenna structure, and has little variation
# 3. UMa has both antenna correlation (via antenna array) and variation

# %%

# fileName = save_model_weights(chanPredictor, modelFileName)
# model = load_model_weights(chanPredictor, modelFileName)