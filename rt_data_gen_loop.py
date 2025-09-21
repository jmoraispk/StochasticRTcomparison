

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepmimo as dm

# To create sequences and videos
from thtt_ch_pred_utils import (
    get_all_sequences,
    process_and_save_channel,
    expand_to_uniform_sequences,
    interpolate_dataset_from_seqs
)

NT = 2
NR = 1

N_SAMPLES = 200_000
L = 60  # 20 for input, 40 for output
N_SUBCARRIERS = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
TIME_DELTA = 1e-3 # [s]

INTERPOLATE = True
INTERP_FACTOR = 10  # final interpolated numbers of points
                     # = number of points between samples - 2 (endpoints)
# Note: if samples are 1m apart, and we want 10cm between points, 
#       set INTERP_FACTOR = 102. 1 m / (102 - 2) = 10cm

GPU_IDX = 0
SEED = 42

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']

dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)


#%%
MAX_DOOPLER = 3 # [Hz]
DATA_FOLDER = f'ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps'

#%% [ANY ENV] 2. Ray tracing data generation: Create sequences

# Sequence length for channel prediction
PRE_INTERP_SEQ_LEN = L if not INTERPOLATE else max(L // INTERP_FACTOR + 1, 2) # min length is 2

all_seqs = get_all_sequences(dataset, min_len=PRE_INTERP_SEQ_LEN)

# Print statistics
seq_lens = [len(seq) for seq in all_seqs]
sum_len_seqs = sum(seq_lens)
avg_len_seqs = sum_len_seqs / len(all_seqs)


#%% [ANY ENV] 3. Ray tracing data generation: Create sequences for Channel Prediction

# Split all sequences in LENGTH L (output: (n_seqs, L)
all_seqs_mat_t = expand_to_uniform_sequences(all_seqs, target_len=PRE_INTERP_SEQ_LEN, stride=1)
print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

# Number of sequences to sample from original sequences
final_samples = min(N_SAMPLES, len(all_seqs_mat_t))
np.random.seed(SEED)
idxs = np.random.choice(len(all_seqs_mat_t), final_samples, replace=False)
all_seqs_mat_t2 = all_seqs_mat_t[idxs][:100] # generate less sequences for testing
print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

#%% [ANY ENV] 4. Ray tracing data generation: Interpolate sequences

if INTERPOLATE:
    dataset_ready = interpolate_dataset_from_seqs(
        dataset,
        all_seqs_mat_t2,
        points_per_segment=INTERP_FACTOR
    )
    print(f"dataset_ready.n_ue: {dataset_ready.n_ue}")
    tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)


#%% [ANY ENV] 5. Ray tracing data generation: Generate channels & Process data

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
ch_params.ofdm.subcarriers = N_SUBCARRIERS
ch_params.ofdm.selected_subcarriers = np.arange(N_SUBCARRIERS)
ch_params.ofdm.bandwidth = 15e3 * N_SUBCARRIERS # [Hz]
ch_params.doppler = True  # Enable doppler computation

# Mean absolute alignment = 1/2 * MAX_DOOPLER
dataset_ready.set_doppler(MAX_DOOPLER / 2) 

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

# Unified post-processing and saving
H_norm, H_noisy_norm, h_max = process_and_save_channel(
    H_complex=H_seq,
    time_axis=1,
    data_folder=DATA_FOLDER,
    model='asu_campus_3p5_10cm' + (f'_interp_{INTERP_FACTOR}' if INTERPOLATE else ''),
    snr_db=SNR
)