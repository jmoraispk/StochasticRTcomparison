"""Utilities for Training and Testing Channel Models

This module brings together a suite of functions designed for the training and 
evaluation of channel models. It includes capabilities for cross-testing among 
various models and visualizing the outcomes. The module is compatible with both 
stochastic and ray tracing channel models.
"""

#%% Import Modules
import sys
sys.path.insert(0, '/mnt/c/Users/jmora/Documents/GitHub/StochasticRTcomparison/')

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = 'data'
N_TAPS = 16
N_ANT = 32
ENCODED_DIM = 32

MATRIX_NAME = 'data_matrices_50k.pkl'
MAT_PATH = os.path.join(DATA_FOLDER, MATRIX_NAME)
DATASET_MAIN_FOLDER = 'channel_datasets_50k'

# Example usage
ch_models = ['CDL-B', 'UMa']
rt_scens = ['asu_campus_3p5']
models = rt_scens + ch_models

#%% Load and Prepare Data

from data_gen import DataConfig, load_data_matrices

# Configure data generation
cfg = DataConfig(
    n_samples = 50_000,
    n_prbs = 20,
    n_rx = 1,
    n_tx = N_ANT,
    snr = 50,
    normalize = 'dataset' # per 'datapoint' or 'dataset'
)

# UMAP parameters
cfg.x_points = cfg.n_samples #int(2e5)  # Number of points to sample from each dataset (randomly)
cfg.seed = 42  # Set to None to keep random
cfg.rt_uniform_steps = [1, 1]

# Load data
data_matrices = load_data_matrices(models, cfg)

#%% Save matrices
os.makedirs(DATA_FOLDER, exist_ok=True)
with open(MAT_PATH, 'wb') as f:
    pickle.dump(data_matrices, f)

#%% Load matrices (in environment with PyTorch)

with open(MAT_PATH, 'rb') as f:
    data_matrices = pickle.load(f)

#%% Train Models

from thtt_utils import (
    train_models,
    cross_test_models,
    plot_training_results,
    plot_test_matrix,
    train_with_percentages
)
# NC=16 * n_ant=32 * 2 -> 32 encoded dim(32x reduction)

# Train models
all_res = train_models(models, data_matrices, DATASET_MAIN_FOLDER, 
                       encoded_dim=ENCODED_DIM, NC=N_TAPS, num_epochs=10)#,
                    #    n_train_samples=10240*20,
                    #    n_val_samples=10000,
                    #    n_test_samples=10000,
                    #    seed=2)
#convert_channel_angle_delay(data_matrices[model])[:,:,:,:NC]

#%% Plot Training Results

plot_training_results(all_res, models, save_path='training_results.png')

#%% Cross-Test Models

all_test_results, results_matrix = \
    cross_test_models(models, data_matrices, DATASET_MAIN_FOLDER, 
                      encoded_dim=ENCODED_DIM, NC=N_TAPS, Nt=N_ANT)

#%% Plot Test Results

plot_test_matrix(results_matrix, models, save_path='test_matrix.png')

results_matrix_db = 10 * np.log10(results_matrix)
df = pd.DataFrame(results_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nTest Results (NMSE in dB):")
print("==========================")
print(df.to_string())

#%% Train with Different Percentages of Data

# Train models with different percentages
os.makedirs(DATASET_MAIN_FOLDER, exist_ok=True)

# Define percentages to use for training
percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Path to pre-trained UMa model
uma_model_path = 'channel_datasets_uma_asu3/model_UMa/model_encoded-dim=128_model=UMa.path'

# Train UMa model with pre-trained weights
uma_test_nmse, uma_name = train_with_percentages(
    'UMa', data_matrices, DATASET_MAIN_FOLDER,
    percentages=percentages,
    uma_model_path=uma_model_path, load_model=True
)

# Train UMa model from scratch
rand_test_nmse, rand_name = train_with_percentages(
    'UMa', data_matrices, DATASET_MAIN_FOLDER,
    percentages=percentages,
    load_model=False
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(percentages, uma_test_nmse, '-o', label='UMA (Pre-trained)')
plt.plot(percentages, rand_test_nmse, '-o', label='Random (From scratch)')
plt.legend()
plt.xlabel('Percentage of Data Used')
plt.ylabel('NMSE (dB)')
plt.title('NMSE of UMA and Random Models')
plt.grid(True)
plt.savefig('percentage_training_results.png')
plt.show() 