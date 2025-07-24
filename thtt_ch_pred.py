#%% Imports

import numpy as np
import os
import matplotlib.pyplot as plt

import deepmimo as dm

#%% Load data

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']
essential_matrices = ['rx_pos', 'aoa_az', 'inter']

dataset = dm.load('asu_campus_3p5_10cm', matrices=essential_matrices)
# dataset = dm.load('asu_campus_3p5', matrices=essential_matrices)

#%% Generate all linear sequences in a scenario

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
    consecutive_arrays = [arr for arr in consecutive_arrays if len(arr) > min_len]
    
    return consecutive_arrays
    
#%% Make video of all sequences

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
            idxs_filtered = idxs[arr]
            plt.scatter(dataset.rx_pos[idxs_filtered, 0], 
                        dataset.rx_pos[idxs_filtered, 1], color='red', s=.5)
        
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

#%% Create all sequences

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

# %%

# TODO: check if stochastic models have time dimension
# TODO: PUT stochastic channel data AND Ray tracing data in the same format (seqs!)

# TODO: add receiver antennas?


#%% [SIONNA ENV] Load and Prepare Data

from tqdm import tqdm
import numpy as np
from data_gen import DataConfig
from sionna_ch_gen import SionnaChannelGenerator
import matplotlib.pyplot as plt


NT = 32
NC = 16

# Configure data generation
data_cfg = DataConfig(
    n_samples = 100_000,
    n_prbs = 20,
    n_rx = 1,
    n_tx = NT, 
    n_time_steps = 20,
    samp_freq = 1e3
)

model = 'TDL-A'
config = data_cfg

print(f"Generating stochastic data for {model}...")
ch_gen = SionnaChannelGenerator(num_prbs=config.n_prbs,
                                channel_name=model,
                                batch_size=config.batch_size,
                                n_rx=config.n_rx,
                                n_tx=config.n_tx,
                                normalize=False,
                                seed=config.seed,
                                ue_speed=10)

#%%

def channel_sample(batch_size=1000, num_time_steps=10, sampling_frequency=1e3):
    """Sample channel coefficients and delays.
    
    Args:
        batch_size (int): Number of samples to generate
        num_time_steps (int): Number of time steps to generate
        sampling_frequency (float): Sampling frequency in Hz
        
    Returns:
        H (np.ndarray): Channel matrix of shape [batch_size, num_rx_ant, num_tx_ant, num_time_steps]
    """
    a, t = ch_gen.channel(batch_size=batch_size, 
                         num_time_steps=num_time_steps, 
                         sampling_frequency=sampling_frequency)

    a, t = a.numpy(), t.numpy()
    # a [batch size, num_rx = 1, num_rx_ant, num_tx=1, num_tx_ant,  num_paths, num_time_steps], tf.complex
    # t [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float) [s]

    # Squeeze num_rx and num_tx dimensions in a & t
    a = a[:, 0, :, 0, :, :, :]  # [batch size, num_rx_ant num_tx_ant, num_paths, num_time_steps]
    t = t[:, 0, 0, :]  # [batch size, num_paths]

    # Calculate phase shifts and sum along paths
    phase_shifts = np.exp(-1j * 2 * np.pi * ch_gen.fc * t)
    H = np.sum(a * phase_shifts[:, None, None, :, None], axis=3)

    return H  # [batch size, num_rx_ant, num_tx_ant, num_time_steps]

H = np.zeros((config.n_samples, data_cfg.n_rx, data_cfg.n_tx, data_cfg.n_time_steps), dtype=np.complex64)

BATCH = 100  # batch supported by Sionna in my computer
pbar = tqdm(range(config.n_samples // BATCH), desc="Generating channel data")
for i in pbar:
    H[i*BATCH:(i+1)*BATCH] = channel_sample(BATCH, config.n_time_steps, config.samp_freq)

print(f"H.shape: {H.shape}")

#%% Prepare data for training

# H.shape: (100000, 3, 32, seq_len)
# Merge antenna dimensions
H2 = H.reshape(H.shape[0], -1, H.shape[-1])
print(f"H2.shape: {H2.shape}") # (batch, n_rx_ant * n_tx_ant, seq_len)

# Put features in the last dimension
H2 = H2.swapaxes(1, 2)
print(f"H2.shape: {H2.shape}") # (batch_size, sequence_length, features)

# Make Real by putting IQ into the features
H2 = np.concatenate([H2.real, H2.imag], axis=2)
print(f"H2.shape: {H2.shape}") # (batch_size, sequence_length, 2*features)

# Normalize (min-max)
H_norm = (H2 - H2.min()) / (H2.max() - H2.min())
print(f"H_norm.shape: {H_norm.shape}")

# Save data
folder = 'ch_pred_data'
os.makedirs(folder, exist_ok=True)
np.save(f'{folder}/H_norm.npy', H_norm) # TODO: add dataset name to file name

#%%
import numpy as np
from nr_channel_predictor_wrapper import construct_model, train, predict, info
import matplotlib.pyplot as plt

H_norm = np.load(f'ch_pred_data/H_norm.npy')

# Split into input/output training/validation sets

def split_data(H_norm: np.ndarray, train_ratio: float = 0.9):
    n_samples = H_norm.shape[0]
    n_train_samples = int(n_samples * train_ratio)
    # IN/OUT: Take the first n-1 time steps as input (x) and the last as output (y)
    # TRAIN/VAL: Take the first samples for training and the last for val.
    x_train = H_norm[:n_train_samples, :-1]
    y_train = H_norm[:n_train_samples, -1]
    x_val = H_norm[n_train_samples:, :-1]
    y_val = H_norm[n_train_samples:, -1]
    print(f"x_train.shape: {x_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"x_val.shape: {x_val.shape}")
    print(f"y_val.shape: {y_val.shape}")
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = split_data(H_norm)

# x_train, y_train, x_val, y_val = split_data(H_norm, l_in=10, l_gap=1)


# L_in = input seq length
# L_gap = gap between input and output (1, 2, 3, ...) [ms]

NT = 32

#%%

# TODO: LOOP ACROSS DIFFERENT HORIZONS
# TODO: LOOP ACROSS DIFFERENT DATASETS (CDL, UMA)
# TODO: LOOP ACROSS DIFFERENT DATASETS (RT)

ch_pred_model = construct_model(NT, hidden_size=128, num_layers=2)

info(ch_pred_model)

trained_model, tr_loss, val_loss, elapsed_time = \
    train(ch_pred_model, x_train, y_train, x_val, y_val, 
          initial_learning_rate=1e-4, batch_size=128, num_epochs=30, 
          validation_freq=1, verbose=True)

# sample & hold baseline
sh_loss = np.mean(np.abs(x_val[:, 0] - y_val))

#%%
plt.plot(10*np.log10(tr_loss), label='Training')
plt.plot(10*np.log10(val_loss), label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss (dB)')
plt.title('Training and validation loss')
plt.legend()
plt.grid()
plt.show()

#%%

y_pred = predict(trained_model, x_val)


#%%

# NOTES:

# 1. TDL has no antenna correlation (unless we introduce it!)
# 2. CDL has ant. corr. but needs the antenna structure, and has little variation
# 3. UMa has both antenna correlation (via antenna array) and variation

# %%
