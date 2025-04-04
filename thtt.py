"""Training and testing utilities for channel model comparison.

This module provides functions for training and testing channel models,
including cross-testing between different models and visualization of results.
It supports both stochastic and ray tracing channel models.
"""

import sys
sys.path.insert(0, '/mnt/c/Users/jmora/Documents/GitHub/StochasticRTcomparison/')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_gen import DataConfig, load_data_matrices
from thtt_utils import (
    train_models,
    cross_test_models,
    plot_training_results,
    plot_test_matrix,
    train_with_percentages
)

#%% Training and Testing Channel Models
# This notebook demonstrates the training and testing of channel models,
# including cross-testing between different models and visualization of results.

#%% Load and Prepare Data
# Example usage
ch_models = ['CDL-B', 'UMa']
rt_scens = ['asu_campus_3p5']
models = rt_scens + ch_models

# Configure data generation
config = DataConfig()
config.n_samples = 10_000
config.n_prbs = 50
config.n_rx = 1
config.n_tx = 32
config.snr = 50

# Load data
data_matrices = load_data_matrices(models, config)

#%% Train Models
# Train models
dataset_main_folder = 'channel_datasets_uma_asu3'
all_res = train_models(models, data_matrices, dataset_main_folder)

#%% Plot Training Results
# Plot training results
plot_training_results(all_res, models, save_path='training_results.png')

#%% Cross-Test Models
# Cross-test models
all_test_results, results_matrix = cross_test_models(models, data_matrices, dataset_main_folder)

#%% Plot Test Results
# Plot test results matrix
plot_test_matrix(results_matrix, models, save_path='test_matrix.png')

# Print test results table
results_matrix_db = 10 * np.log10(results_matrix)
df = pd.DataFrame(results_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nTest Results (NMSE in dB):")
print("==========================")
print(df.to_string())

#%% Train with Different Percentages of Data

# Train models with different percentages
dataset_main_folder = 'channel_datasets_uma_asu_percentages'
os.makedirs(dataset_main_folder, exist_ok=True)

# Define percentages to use for training
percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Path to pre-trained UMa model
uma_model_path = 'channel_datasets_uma_asu3/model_UMa/model_encoded-dim=128_model=UMa.path'

# Train UMa model with pre-trained weights
uma_test_nmse, uma_name = train_with_percentages(
    'UMa', data_matrices, dataset_main_folder,
    percentages=percentages,
    uma_model_path=uma_model_path, load_model=True
)

# Train UMa model from scratch
rand_test_nmse, rand_name = train_with_percentages(
    'UMa', data_matrices, dataset_main_folder,
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