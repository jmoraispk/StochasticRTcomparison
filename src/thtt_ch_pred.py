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
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepmimo as dm

# To create sequences and videos
from thtt_ch_pred_utils import (
    get_all_sequences,
    make_sequence_video,
    db,
    nmse,
    process_and_save_channel,
    split_data,
    expand_to_uniform_sequences,
    interpolate_dataset_from_seqs
)

# To plot test matrix
from thtt_plot import plot_test_matrix

# To plot H for specific antennas (uses only matplotlib)
from thtt_ch_pred_plot import plot_iq_from_H, plot_validation_losses_from_csv
from thtt_ch_pred_utils import compute_nmse_matrix

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

DATA_FOLDER = f'ch_pred_results/ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps'

GPU_IDX = 0
SEED = 42

#%% [PYTORCH ENVIRONMENT] Split data

from nr_channel_predictor_wrapper import (
    construct_model, train, predict, info, 
    save_model_weights, load_model_weights
)

import pandas as pd

models_folder = f'ch_pred_results/FINAL_ch_pred_models_{MAX_DOOPLER}hz_{L}steps_INTERP_{INTERP_FACTOR}'
os.makedirs(models_folder, exist_ok=True)

models = ['TDL-A', 'CDL-C', 'UMa', f'asu_campus_3p5_10cm_interp_{INTERP_FACTOR}']

L_IN = 20  # input sequence length

#%% [PYTORCH ENVIRONMENT] Train models

horizons = [1, 3, 5, 10, 20, 40]

val_loss_per_horizon_gru = {model: [] for model in models}
val_loss_per_horizon_gru_best = {model: [] for model in models}
val_loss_per_horizon_sh = {model: [] for model in models}

for model in models:
    H_norm = np.load(f'{DATA_FOLDER}/H_norm_{model}.npy') # (n_samples, seq_len)

    for horizon in horizons:
        print(f"========== Horizon: {horizon} ==========")

        model_weights_file = f'{models_folder}/{model}_{horizon}.pth'
        if os.path.exists(model_weights_file):
            print(f"Model weights file {model_weights_file} already exists. Skipping training.")
            continue

        x_train, y_train, x_val, y_val = split_data(H_norm, l_in=L_IN, l_gap=horizon)

        ch_pred_model = construct_model(NT, hidden_size=128, num_layers=3)
        
        info(ch_pred_model)

        trained_model, tr_loss, val_loss, elapsed_time = \
            train(ch_pred_model, x_train, y_train, x_val, y_val, 
                initial_learning_rate=4e-4, batch_size=256, num_epochs=300, 
                verbose=True, patience=60, patience_factor=1,
                best_model_path=model_weights_file.replace('.pth', '_best.pth'),
                device_idx=GPU_IDX)

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

# Save validation loss results to CSV
results = {'horizon': horizons}

# Add results for each model
for model in models:
    results[f'{model}_gru'] = val_loss_per_horizon_gru[model]
    results[f'{model}_gru_best'] = val_loss_per_horizon_gru_best[model]
    results[f'{model}_sh'] = val_loss_per_horizon_sh[model]

# Convert to DataFrame and save
df = pd.DataFrame(results)
df.to_csv(f'{models_folder}/validation_losses.csv', index=False)
print(f"Saved validation loss results to {models_folder}/validation_losses.csv")


#%% Plot validation loss per horizon results

# Load validation loss results from CSV
# df = pd.read_csv(f'{models_folder}/validation_losses_final.csv')
df = pd.read_csv(f'{models_folder}/validation_losses-final.csv')

# Extract horizons and per-model losses from the DataFrame
horizons = df['horizon'].tolist()

# Initialize dictionaries to store the loaded losses
val_loss_per_horizon_gru = {}
val_loss_per_horizon_gru_best = {}
val_loss_per_horizon_sh = {}

for model in models:
    val_loss_per_horizon_gru[model] = df[f'{model}_gru'].tolist()
    val_loss_per_horizon_gru_best[model] = df[f'{model}_gru_best'].tolist()
    val_loss_per_horizon_sh[model] = df[f'{model}_sh'].tolist()


o = 0 # index of first horizon to plot (in case we want to start from non-zero)
plt.figure(dpi=200)
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # Colorblind-friendly palette
markers = ['o', 's', 'D', 'P']
for i, model in enumerate(models):
    # plt.plot(horizons[o:], val_loss_per_horizon_gru[model][o:], label=model, 
    #          color=colors[i], marker=markers[i], markersize=5)
    plt.plot(horizons[o:], val_loss_per_horizon_gru_best[model][o:], label=model + '_best', 
             color=colors[i], marker=markers[i], markersize=5)
    plt.plot(horizons[o:], val_loss_per_horizon_sh[model][o:], label=model + '_SH', 
             color=colors[i], linestyle='--', marker=markers[i], markersize=5)
plt.xlabel('Horizon (ms)')
plt.ylabel('Validation Loss (NMSE in dB)')
plt.legend(ncols=4, bbox_to_anchor=(0.46, 1.0), loc='lower center')
plt.xlim(0, horizons[-1] + 0.5)
plt.grid()
plt.savefig(f'{models_folder}/validation_losses.png', bbox_inches='tight', dpi=200)
plt.show()

#%% Load models and test them on other datasets

# Run cross-testing over all defined models
results_matrix = compute_nmse_matrix(models, horizon=5, l_in=L_IN,
                                     models_folder=models_folder,
                                     data_folder=DATA_FOLDER,
                                     num_tx_antennas=NT)

# Plot confusion matrix (NMSE in dB inside the function)
# plot_test_matrix(results_matrix, models)
asu_name = f'ASU-{100 // INTERP_FACTOR}mm' if INTERP_FACTOR != 2 else 'ASU-40mm'
plot_test_matrix(results_matrix, ['TDL-A', 'CDL-C', 'UMa', asu_name])

#%% Fine tuning models & evaluating performance on target datasets

# Fine-tuning configuration and utilities
finetuned_models_folder = models_folder + '_finetuned'
os.makedirs(finetuned_models_folder, exist_ok=True)

def predict_batched(model, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    preds = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        preds.append(predict(model, x[start:end]))
    return np.concatenate(preds, axis=0)


def fine_tune_and_test(models_list: list[str], horizon: int, l_in: int,
                       train_ratio: float = 0.9,
                       initial_lr: float = 4e-4,
                       batch_size: int = 256,
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
        base_weights = f"{models_folder}/{src_model}_{horizon}_best.pth"
        print(f"\n[Fine-tune] Using base weights: {base_weights}")

        for j, tgt_model in enumerate(models_list):
            print(f"Fine-tuning {src_model} -> {tgt_model} (horizon={horizon})")

            if src_model == tgt_model:
                print(f"Skipping fine-tuning {src_model} -> {tgt_model} (horizon={horizon}) because it's the same model")
                continue

            # Load target data and split
            H_norm_tgt = np.load(f'{DATA_FOLDER}/H_norm_{tgt_model}.npy')
            x_train, y_train, x_val, y_val = split_data(H_norm_tgt, train_ratio=train_ratio,
                                                        l_in=l_in, l_gap=horizon)

            # Load model from base weights
            model = construct_model(NT, hidden_size=128, num_layers=3)
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
ft_matrix = fine_tune_and_test(models, horizon=5, l_in=L_IN,
                               train_ratio=0.01,
                               initial_lr=4e-4,
                               batch_size=128,
                               num_epochs=30,
                               patience=10)

plot_test_matrix(ft_matrix, models)

#%%

from pathlib import Path

base_dir = Path(".").resolve() / "ch_pred_results"
folder = base_dir / "FINAL_ch_pred_models_100hz_60steps_INTERP_10"
csv_path = folder / "validation_losses-final.csv"

out_path = folder / "validation_losses.png"
plot_validation_losses_from_csv(csv_path, out_path, split_legend=True)
