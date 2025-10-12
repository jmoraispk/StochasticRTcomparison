"""
[Data generation for channel prediction]

Ray tracing data generation (DeepMIMO):
1. Load data
2. Create all sequences of samples in RT scenario (list of arrays)
3. Create uniform (same length) sequences for Channel Prediction (N seq x L array)
4. (optional) Interpolate sequences (list of arrays with K times the length)
5. Generate channels for all points in the sequences
6. Post-process data (add noise, normalize, save)

Stochastic data generation (Sionna):
1. Import and create data generator
2. Generate channel data (N seq x L array)
6. Post-process data (add noise, normalize, save)

Post-processing (common for ray tracing & stochastic):
1. Reshape to (n_samples, seq_len, features)
2. Concatenate real & imaginary parts (n_samples, seq_len, real_features)
3. Add noise (based on SNR)
4. Normalize (by max absolute value of real features)
5. Save data (if save=True, default)

"""


#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepmimo as dm

# To create sequences and videos
from thtt_ch_pred_utils import (
    get_all_sequences,
    make_sequence_video,  # noqa: F401
    process_and_save_channel,
    expand_to_uniform_sequences,
    interpolate_dataset_from_seqs
)

# To plot H for specific antennas (uses only matplotlib)
from thtt_ch_pred_plot import plot_iq_from_H

NT = 2
NR = 1

N_SAMPLES = 200_000
L = 60  # 20 for input, 40 for output
N_SUBCARRIERS = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
MAX_DOOPLER = 100 # [Hz]
TIME_DELTA = 1e-3 # [s]

INTERPOLATE = True
INTERP_FACTOR = 10  # final interpolated numbers of points
                     # = number of points between samples - 2 (endpoints)
# Note: if samples are 1m apart, and we want 10cm between points, 
#       set INTERP_FACTOR = 102. 1 m / (102 - 2) = 10cm

DATA_FOLDER = f'../data/ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps'

GPU_IDX = 0
SEED = 42

#%% [ANY ENV] 1. Ray tracing data generation: Load data

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']

dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)
# dataset = dm.load('asu_campus_3p5', matrices=matrices)

#%% [ANY ENV] (optional) Ray tracing data: Make video of all sequences

# make_sequence_video(dataset, folder='sweeps', ffmpeg_fps=60)

#%% [ANY ENV] 2. Ray tracing data generation: Create sequences

# Sequence length for channel prediction
PRE_INTERP_SEQ_LEN = L if not INTERPOLATE else max(L // INTERP_FACTOR + 1, 2) # min length is 2
# Note: interpolation will scale the sequence length by INTERP_FACTOR to be >= L
#       (at least 2 samples needed for interpolation)

# RT sample distance / INTERP_FACTOR = sample distance in the interpolated dataset

all_seqs = get_all_sequences(dataset, min_len=PRE_INTERP_SEQ_LEN)

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

dataset_ready = dataset

#%% [ANY ENV] 3. Ray tracing data generation: Create sequences for Channel Prediction

# Split all sequences in LENGTH L (output: (n_seqs, L)
all_seqs_mat_t = expand_to_uniform_sequences(all_seqs, target_len=PRE_INTERP_SEQ_LEN, stride=1)
print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

# Number of sequences to sample from original sequences
final_samples = min(N_SAMPLES, len(all_seqs_mat_t))
np.random.seed(SEED)
idxs = np.random.choice(len(all_seqs_mat_t), final_samples, replace=False)
all_seqs_mat_t2 = all_seqs_mat_t[idxs] # [:100] generate less sequences for testing
print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

#%% [ANY ENV] 4. Ray tracing data generation: Interpolate sequences

# When we interpolate, it's easier to sample sequences of length L from the original sequences, 
# and then create a dataset with only the interpolated results.

# If we don't interpolate, we can sample sequences of length L from the interpolated dataset.

if INTERPOLATE:
    dataset_ready = interpolate_dataset_from_seqs(
        dataset,
        all_seqs_mat_t2,
        points_per_segment=INTERP_FACTOR
    )
    print(f"dataset_ready.n_ue: {dataset_ready.n_ue}")
    tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)

# from thtt_ch_pred_utils import interpolate_dataset_from_seqs_old

# if INTERPOLATE:
#     dataset_ready_old = interpolate_dataset_from_seqs_old(
#         dataset,
#         all_seqs_mat_t2,
#         points_per_segment=INTERP_FACTOR
#     )
#     print(f"dataset_ready_old.n_ue: {dataset_ready_old.n_ue}")
#     tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)

#%% [ANY ENV] 5. Ray tracing data generation: Generate channels & Process data

# NOTE: when the product seq_len * n_seqs >> n_ue, it's better to generate channels first
#       and then take the sequences from the generated channels, because channel gen
#       is the most expensive part of the data generation process.
#       IF, instead, the product seq_len * n_seqs < n_ue, it's better to generate sequences first, 
#       trim the dataset to the necessary users, and then generate channels for the users in the sequences.
# TODO: implement this choice when selecting data...
# Currently: without interpolation, we gen channels first and select channels from sequence indices after. 
#            with interpolation, we select sequences and trim the dataset to the necessary users.
#            (with interpolation, this necessary because NEW points are in the dataset)

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
ch_params.ofdm.subcarriers = N_SUBCARRIERS
ch_params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
ch_params.ofdm.bandwidth = 15e3 * N_SUBCARRIERS # [Hz]
ch_params.doppler = True  # Enable doppler computation

doppler_way = 1

# Way 1 of adding doppler: same doppler to all users / paths
if doppler_way == 1:
    # Mean absolute alignment = 1/2 * MAX_DOOPLER
    dataset_ready.set_doppler(MAX_DOOPLER / 2)  # Add the same doppler to all users / paths

# Way 2 of adding doppler: different doppler per user / path assuming const. speed & direction
if doppler_way == 2:
    dataset_ready.rx_vel = np.array([10, 0, 0]) # [m/s] along x-axis

# Way 3 of adding doppler: different doppler per user / path assuming const. speed, 
#                          with direction derived from the path geometry
if doppler_way == 3:
    dataset_ready.rx_vel = np.array([10, 0, 0]) # [m/s] along x-axis
    # TODO!

# Way 4 of adding doppler: different doppler per user / path deriving speed & direction 
#                          from the path geometry
if doppler_way == 4:
    pass # not necessary here: speeds are constant
    # when we have uniform sampling & interpolation

H = dataset_ready.compute_channels(ch_params, times=np.arange(L) * TIME_DELTA)
print(f"H.shape: {H.shape}")  # (n_samples * L, NR, NT, N_SUBCARRIERS, L)

n_seqs = all_seqs_mat_t2.shape[0]
H_seq = np.zeros((n_seqs, L, NR, NT, N_SUBCARRIERS), dtype=np.complex64)

# Take sequences
for seq_idx in tqdm(range(n_seqs), desc="Taking sequences"):
    for sample_idx_in_seq in range(L):
        if INTERPOLATE:
            idx_in_h = seq_idx * L + sample_idx_in_seq
        else:
            idx_in_h = all_seqs_mat_t2[seq_idx, sample_idx_in_seq]
        H_seq[seq_idx, sample_idx_in_seq] = H[idx_in_h, ..., sample_idx_in_seq]
    # For each sequence, take the channels for the corresponding time steps
    # e.g. first sample of sequence is at time 0, last sample of sequence is at time L-1

print(f"H_seq.shape: {H_seq.shape}") # (n_samples, seq_len, n_rx_ant, n_tx_ant, subcarriers)

# Plot H - transform into (n_samples, n_rx_ant, n_tx_ant, seq_len)
H_3_plot = np.transpose(H_seq[:, :, :, :, 0], (0, 2, 3, 1))
plot_sample_idx, plot_rx_idx = plot_iq_from_H(H_3_plot)

# Unified post-processing and saving
H_norm, H_noisy_norm, h_max = process_and_save_channel(
    H_complex=H_seq,
    time_axis=1,
    data_folder=DATA_FOLDER,
    model='asu_campus_3p5_10cm' + (f'_interp_{INTERP_FACTOR}' if INTERPOLATE else ''),
    snr_db=SNR
)

# Plot normalized version
plot_iq_from_H(H_3_plot / h_max, plot_sample_idx, plot_rx_idx)

#%% [SIONNA ENV] Stochastic data generation: Import and Create Data Generator

from tqdm import tqdm
from data_gen import DataConfig
from sionna_ch_gen import SionnaChannelGenerator

# Configure data generation
data_cfg = DataConfig(
    n_samples = N_SAMPLES,
    n_prbs = 20,
    n_rx = NR,
    n_tx = NT, 
    n_time_steps = L,
    samp_freq = 1e3,
    batch_size = 1_000,
    seed = SEED
)

models = ['TDL-A', 'CDL-C', 'UMa']   # models to generate data for

config = data_cfg

fc = 3.5e9 # [Hz]
speed = MAX_DOOPLER / (fc / 3e8) # E.g. at 3 m/s & 3.5 GHz, max Doppler = 37 Hz

def channel_sample(ch_gen, model, batch_size=1000, num_time_steps=10, sampling_frequency=1e3):
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

    H = a.sum(axis=3)

    return H  # [batch size, num_rx_ant, num_tx_ant, num_time_steps]

for model in models:
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

    # Generate channel data
    H = np.zeros((config.n_samples, data_cfg.n_rx, data_cfg.n_tx, data_cfg.n_time_steps), dtype=np.complex64)

    b = config.batch_size
    pbar = tqdm(range(config.n_samples // b), desc="Generating channel data")
    for i in pbar:
        H[i*b:(i+1)*b] = channel_sample(ch_gen, model, b, config.n_time_steps, config.samp_freq)

    print(f"H.shape: {H.shape}")

    # Plot H for specific antennas 
    plot_sample_idx, plot_rx_idx = plot_iq_from_H(H)

    # Unified post-processing and saving
    H_norm, H_noisy_norm, h_max = process_and_save_channel(
        H_complex=H,
        time_axis=-1,
        data_folder=DATA_FOLDER,
        model=model,
        snr_db=SNR
    )

    plot_iq_from_H(H / h_max, plot_sample_idx, plot_rx_idx)
