#%% [MANDATORY CONSTANTS - ANY ENV] Imports

import numpy as np
import os
import matplotlib.pyplot as plt

import deepmimo as dm

# To plot H for specific antennas (uses only matplotlib)
from thtt_ch_pred_plot import plot_iq_from_H

NT = 2
NR = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise
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

matrices = ['rx_pos', 'tx_pos', 'aoa_az', 'aod_az', 'aoa_el', 'aod_el', 
            'delay', 'power', 'phase', 'inter']
# essential_matrices = ['rx_pos', 'aoa_az', 'inter']

dataset = dm.load('asu_campus_3p5_10cm', matrices=matrices)
# dataset = dm.load('asu_campus_3p5')#, matrices=essential_matrices)

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

# TODO: compute channels only for users in all_seqs_mat_t2

# Create channels
ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [NT, 1]
ch_params.ue_antenna.shape = [NR, 1]
dataset.set_doppler(MAX_DOOPLER)
H = dataset.compute_channels(ch_params)
print(f"H.shape: {H.shape}")

# Take sequences right away (some data may repeat, particularily for short sequences)

H_seq = H[all_seqs_mat_t2, ...]
print(f"H_seq.shape: {H_seq.shape}") # (n_samples, seq_len, n_rx_ant, n_tx_ant, subcarriers)

# Plot H - transform to fit: (n_samples, n_rx_ant, n_tx_ant, seq_len)
H_3_plot = np.transpose(H_seq[:, :, :, :, 0], (0, 2, 3, 1))
plot_sample_idx, plot_rx_idx = plot_iq_from_H(H_3_plot)

# Concatenate feature dimensions (rx ant, tx ant, subcarriers)
H2 = H_seq.reshape(H_seq.shape[0], H_seq.shape[1], -1)
print(f"H2.shape: {H2.shape}") # (n_samples, seq_len, features = n_rx_ant * n_tx_ant * subcarriers)

# Make Real by putting IQ into the features
H3 = np.concatenate([H2.real, H2.imag], axis=-1)
print(f"H3.shape: {H3.shape}") # (n_samples, seq_len, 2*features)

# Add noise
noise_var = 10**(-SNR/10)
noise = np.random.randn(*H3.shape) * np.sqrt(noise_var)
H_noisy = H3 + noise

# Normalize (take abs to account for negative values - not because of IQ)
h_max = np.nanmax(np.abs(H_noisy))  # Note: for SNR < 100, max h_norm != max(h_noisy)
H_noisy_norm = H_noisy / h_max
H_norm = H3 / h_max
print(f"H_norm.shape: {H_norm.shape}") # (n_samples, features)
print(f"H_noisy_norm.shape: {H_noisy_norm.shape}") # (n_samples, features)

# Plot normalized version
plot_iq_from_H(H_3_plot / h_max, plot_sample_idx, plot_rx_idx)

# Save data
model = 'asu_campus_3p5'
folder = 'ch_pred_data'
os.makedirs(folder, exist_ok=True)
np.save(f'{folder}/H_norm_{model}.npy', H_norm.astype(np.float32))
np.save(f'{folder}/H_noisy_norm_{model}.npy', H_noisy_norm.astype(np.float32))

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

model = 'UMa'
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
print(f"H2.shape: {H2.shape}") # (n_samples, n_rx_ant * n_tx_ant, seq_len)

# Put features in the last dimension
H2 = H2.swapaxes(1, 2)
print(f"H2.shape: {H2.shape}") # (n_samples, sequence_length, features)

# Make Real by putting IQ into the features
H2 = np.concatenate([H2.real, H2.imag], axis=2)
print(f"H2.shape: {H2.shape}") # (n_samples, sequence_length, 2*features)

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

models_folder = 'ch_pred_models2'
os.makedirs(models_folder, exist_ok=True)

models = ['TDL-A', 'CDL-C', 'UMa', 'asu_campus_3p5']
# models = ['TDL-A', 'CDL-C', 'UMa']
# models = ['asu_campus_3p5']
# models = ['TDL-A']

horizons = [1, 3, 5, 10, 20, 40]
L = 10  # input sequence length

val_loss_per_horizon_gru = {model: [] for model in models}
val_loss_per_horizon_gru_best = {model: [] for model in models}
val_loss_per_horizon_sh = {model: [] for model in models}

for model in models:
    H_norm = np.load(f'ch_pred_data/H_norm_{model}.npy') # (n_samples, seq_len)

    for horizon in horizons:
        print(f"========== Horizon: {horizon} ==========")

        model_weights_file = f'{models_folder}/{model}_{horizon}.pth'
        # if os.path.exists(model_weights_file):
        #     print(f"Model weights file {model_weights_file} already exists. Skipping training.")
        #     continue

        x_train, y_train, x_val, y_val = split_data(H_norm, l_in=L, l_gap=horizon)

        ch_pred_model = construct_model(NT, hidden_size=128, num_layers=2)
        
        info(ch_pred_model)

        trained_model, tr_loss, val_loss, elapsed_time = \
            train(ch_pred_model, x_train, y_train, x_val, y_val, 
                initial_learning_rate=1e-4, batch_size=64, num_epochs=280, 
                verbose=True, patience=30, patience_factor=1,
                best_model_path=f'{models_folder}/{model}_{horizon}_best.pth')

        save_model_weights(trained_model, model_weights_file)

        # Plot training and validation loss
        plt.plot(db(tr_loss), label='Training')
        plt.plot(db(val_loss), label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('NMSE Loss (dB)')
        plt.title(f'Training and validation loss for {model} horizon {horizon} ms')
        plt.legend()
        plt.grid()
        plt.savefig(f'{models_folder}/{model}_{horizon}_loss.png', bbox_inches='tight', dpi=200)
        plt.close()
        
        # sample & hold baseline (i.e. no prediction)
        sh_loss = nmse(x_val[:, -1], y_val)

        val_loss_per_horizon_gru[model].append(db(val_loss[-1]))
        val_loss_per_horizon_gru_best[model].append(db(min(val_loss)))
        val_loss_per_horizon_sh[model].append(db(sh_loss))
        
        print(f"Final validation loss for {model} horizon {horizon} ms:")
        print(f"  GRU: {db(val_loss[-1]):.1f} dB")
        print(f"  GRU best: {db(min(val_loss)):.1f} dB")
        print(f"  S&H: {db(sh_loss):.1f} dB")

            

#%% Plot validation loss per horizon results

o = 0 # index of first horizon to plot (in case we want to start from non-zero)
plt.figure(dpi=200)
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Colorblind-friendly palette
markers = ['o', 's', 'D', 'P']
for i, model in enumerate(models):
    plt.plot(horizons[o:], val_loss_per_horizon_gru[model][o:], label=model, 
             color=colors[i], marker=markers[i], markersize=5)
    plt.plot(horizons[o:], val_loss_per_horizon_gru_best[model][o:], label=model + '_best', 
             color=colors[i], linestyle='-.', marker=markers[i], markersize=5)
    plt.plot(horizons[o:], val_loss_per_horizon_sh[model][o:], label=model + '_SH', 
             color=colors[i], linestyle='--', marker=markers[i], markersize=5)
plt.xlabel('Horizon (ms)')
plt.ylabel('Validation Loss (NMSE in dB)')
plt.legend(ncols=4, bbox_to_anchor=(0.46, 1.0), loc='lower center')
plt.xlim(0, horizons[-1] + 0.5)
plt.grid()
plt.show()

#%% Load models and test them on other datasets

from thtt_plot import plot_test_matrix

# Cross-testing configuration
models_folder_eval = 'ch_pred_models2'

# Ensure input length L used for evaluation matches training
# Uses the same L already defined above


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


def compute_nmse_matrix(models_list: list[str], horizon: int, l_in: int) -> np.ndarray:
    """Load each trained model and evaluate NMSE on every dataset."""
    results = np.zeros((len(models_list), len(models_list)), dtype=float)

    for i, src_model in enumerate(models_list):
        weights_path = f"{models_folder_eval}/{src_model}_{horizon}.pth"
        print(f"\nLoading model weights: {weights_path}")

        ch_pred_model = construct_model(NT, hidden_size=128, num_layers=2)
        ch_pred_model = load_model_weights(ch_pred_model, weights_path)

        for j, tgt_model in enumerate(models_list):
            print(f"Testing {src_model} on {tgt_model} (horizon={horizon})")
            H_norm_tgt = np.load(f'ch_pred_data/H_norm_{tgt_model}.npy')
            x_all, y_all = build_xy_all(H_norm_tgt, l_in=l_in, l_gap=horizon)

            # Batched inference to prevent GPU OOM
            batch_size = 128
            preds = []
            for start in range(0, x_all.shape[0], batch_size):
                end = min(start + batch_size, x_all.shape[0])
                y_pred_b = predict(ch_pred_model, x_all[start:end])
                preds.append(y_pred_b)
            y_pred = np.concatenate(preds, axis=0)

            results[i, j] = nmse(y_pred, y_all)

            print(f"  NMSE: {10*np.log10(results[i, j]):.1f} dB")

    return results


# Run cross-testing over all defined models
results_matrix = compute_nmse_matrix(models, horizon=1, l_in=10)

# Plot confusion matrix (NMSE in dB inside the function)
plot_test_matrix(results_matrix, models)

#%% Fine tuning models & evaluating performance on target datasets

# Fine-tuning configuration and utilities
finetuned_models_folder = 'ch_pred_models_finetuned'
os.makedirs(finetuned_models_folder, exist_ok=True)


def predict_batched(model, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    preds = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        preds.append(predict(model, x[start:end]))
    return np.concatenate(preds, axis=0)


def fine_tune_and_test(models_list: list[str], horizon: int, l_in: int,
                       train_ratio: float = 0.9,
                       initial_lr: float = 1e-4,
                       batch_size: int = 64,
                       num_epochs: int = 80,
                       patience: int = 15,
                       patience_factor: float = 1.0) -> np.ndarray:
    """Fine-tune each source model on every target dataset and evaluate on target.

    Saves weights to finetuned_models_folder as '{src}_to_{tgt}_{horizon}.pth'.
    Returns an NMSE matrix where [i, j] is the performance of src=i fine-tuned on tgt=j,
    evaluated on the tgt validation split.
    """
    ft_results = np.zeros((len(models_list), len(models_list)), dtype=float)

    for i, src_model in enumerate(models_list):
        base_weights = f"{models_folder_eval}/{src_model}_{horizon}.pth"
        print(f"\n[Fine-tune] Using base weights: {base_weights}")

        for j, tgt_model in enumerate(models_list):
            print(f"Fine-tuning {src_model} -> {tgt_model} (horizon={horizon})")

            # Load target data and split
            H_norm_tgt = np.load(f'ch_pred_data/H_norm_{tgt_model}.npy')
            x_train, y_train, x_val, y_val = split_data(H_norm_tgt, train_ratio=train_ratio,
                                                        l_in=l_in, l_gap=horizon)

            # Load model from base weights
            model = construct_model(NT, hidden_size=128, num_layers=2)
            model = load_model_weights(model, base_weights)

            # Train further on target data (fine-tune)
            model, tr_loss, val_loss, elapsed_time = train(
                model, x_train, y_train, x_val, y_val,
                initial_learning_rate=initial_lr,
                batch_size=batch_size,
                num_epochs=num_epochs,
                verbose=True,
                patience=patience,
                patience_factor=patience_factor,
                best_model_path=f"{finetuned_models_folder}/{src_model}_to_{tgt_model}_{horizon}_best.pth"
            )

            # Save fine-tuned model
            ft_path = f"{finetuned_models_folder}/{src_model}_to_{tgt_model}_{horizon}.pth"
            save_model_weights(model, ft_path)

            # Evaluate on validation split (acts as held-out test here)
            y_pred = predict_batched(model, x_val, batch_size=128)
            nmse_val = nmse(y_pred, y_val)
            ft_results[i, j] = nmse_val
            print(f"  Fine-tuned NMSE on {tgt_model}: {10*np.log10(nmse_val):.1f} dB")

    return ft_results


# Run fine-tuning and plot results
ft_matrix = fine_tune_and_test(models, horizon=1, l_in=10,
                               train_ratio=0.1,
                               initial_lr=1e-4,
                               batch_size=64,
                               num_epochs=50,
                               patience=30)

plot_test_matrix(ft_matrix, models)

