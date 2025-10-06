#%% Imports

import numpy as np
import os
import matplotlib.pyplot as plt
from thtt_ch_pred_utils import db, nmse, split_data
from nr_channel_predictor_wrapper import construct_model, train, info, save_model_weights
import pandas as pd

NT = 2
NR = 1

N_SAMPLES = 200_000
L = 60  # 20 for input, 40 for output
L_IN = 20  # input sequence length
N_SUBCARRIERS = 1

SNR = 250 # [dB] NOTE: for RT, normalization must be consistent for w & w/o noise

TIME_DELTA = 1e-3 # [s]

INTERPOLATE = True

GPU_IDX = 0

# dopplers = [10, 100, 400]
dopplers = [0, 10, 30, 100, 200]
interps = [10]
horizons = [1, 3, 5, 10, 20, 40]

# base_folder = '.'
base_folder = '/home/joao/Downloads/all_ch_pred_RT_data'

for doppler in dopplers:
    for interp in interps:
        MAX_DOOPLER = doppler
        INTERP_FACTOR = interp
        DATA_FOLDER = f'{base_folder}/ch_pred_data_{N_SAMPLES//1000}k_{MAX_DOOPLER}hz_{L}steps'
        models_folder = f'new_rt_ch_pred_models_{MAX_DOOPLER}hz_{L}steps_INTERP_{INTERP_FACTOR}'
        os.makedirs(models_folder, exist_ok=True)
    
        models = [f'asu_campus_3p5_10cm_interp_{INTERP_FACTOR}']

        val_loss_per_horizon_gru = {model: [] for model in models}
        val_loss_per_horizon_gru_best = {model: [] for model in models}
        val_loss_per_horizon_sh = {model: [] for model in models}

        for model in models:
            H_norm = np.load(f'{DATA_FOLDER}/H_norm_{model}.npy') # (n_samples, seq_len)

            for horizon in horizons:
                print(f"========== Horizon: {horizon} ==========")

                model_weights_file = f'{models_folder}/{model}_{horizon}.pth'
                #if os.path.exists(model_weights_file):
                #    print(f"Model weights file {model_weights_file} already exists. Skipping training.")
                #    continue

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
        csv_filename = 'validation_losses3.csv'
        df.to_csv(f'{models_folder}/{csv_filename}', index=False)
        print(f"Saved validation loss results to {models_folder}/{csv_filename}")

# %%
