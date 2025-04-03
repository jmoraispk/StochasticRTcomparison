"""Training and testing utilities for channel model comparison.

This module provides functions for training and testing channel models,
including cross-testing between different models and visualization of results.
It supports both stochastic and ray tracing channel models.
"""

import numpy as np
import pandas as pd

from data_gen import DataConfig, load_data_matrices
from thtt_utils import (
    train_models,
    cross_test_models,
    plot_training_results,
    plot_test_matrix
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