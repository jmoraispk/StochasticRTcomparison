

#%% Imports
import gc

import numpy as np
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

N_SAMPLES = 400_000
L = 60  # 20 for input, 40 for output
N_SUBCARRIERS = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
TIME_DELTA = 1e-3 # [s]

INTERPOLATE = True

GPU_IDX = 0
SEED = 42

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']

#%%

# DOPPLERS = [0, 10, 30, 100, 200]
# INTERPOLATIONS = [5, 10, 30, 100]
DOPPLERS = [10] # NEXT: 30
INTERPOLATIONS = [100] # NEXT: 5

for doppler in DOPPLERS:
    for interp in INTERPOLATIONS:
        print(f"Doppler: {doppler}, Interp: {interp}")

        dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)

        gc.collect()

        MAX_DOOPLER = doppler # [Hz]
        INTERP_FACTOR = interp

        # Ray tracing data generation: Create sequences

        # Sequence length for channel prediction
        PRE_INTERP_SEQ_LEN = L if not INTERPOLATE else max(L // INTERP_FACTOR + 1, 2) # min length is 2

        all_seqs = get_all_sequences(dataset, min_len=PRE_INTERP_SEQ_LEN)

        #  Ray tracing data generation: Create sequences for Channel Prediction

        # Split all sequences in LENGTH L (output: (n_seqs, L)
        all_seqs_mat_t = expand_to_uniform_sequences(all_seqs, target_len=PRE_INTERP_SEQ_LEN, stride=1)
        print(f"all_seqs_mat_t.shape: {all_seqs_mat_t.shape}")

        # Number of sequences to sample from original sequences
        final_samples = min(N_SAMPLES, len(all_seqs_mat_t))
        np.random.seed(SEED)
        idxs = np.random.choice(len(all_seqs_mat_t), final_samples, replace=False)
        all_seqs_mat_t2 = all_seqs_mat_t[idxs]#[:100] # generate less sequences for testing
        print(f"all_seqs_mat_t2.shape: {all_seqs_mat_t2.shape}")

        # Ray tracing data generation: Interpolate sequences

        if INTERPOLATE:
            dataset_ready = interpolate_dataset_from_seqs(
                dataset,
                all_seqs_mat_t2,
                points_per_segment=INTERP_FACTOR
            )
            print(f"dataset_ready.n_ue: {dataset_ready.n_ue}")
            tgt_shape = (all_seqs_mat_t2.shape[0], -1, NR, NT, 1)


        # Ray tracing data generation: Generate channels & Process data

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
        H_norm, _, h_max = process_and_save_channel(
            H_complex=H_seq,
            time_axis=1,
            data_folder=f'all_ch_pred_RT_data/ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps',
            model='asu_campus_3p5_10cm' + (f'_interp_{INTERP_FACTOR}' if INTERPOLATE else ''),
            snr_db=SNR
        )
# %%
