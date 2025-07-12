"""Utilities for Training and Testing Channel Models

This module brings together a suite of functions designed for the training and 
evaluation of channel models. It includes capabilities for cross-testing among 
various models and visualizing the outcomes. The module is compatible with both 
stochastic and ray tracing channel models.

Run instructions:
1. Use the sionna enviornment to run the first 3 cells (3rd cell saved data)
2. Use the pytorch environment to run the 1st cell (imports), the 4th cell (loads data),
   and cells after that. (e.g. 5th cell trains models, 6th plots training results, etc.)
"""

#%% Import Modules
import os
import pickle
import numpy as np
import pandas as pd

# Data paths
DATA_FOLDER = 'data'
MATRIX_NAME = 'data_matrices_10k_complex.pkl'
MAT_PATH = os.path.join(DATA_FOLDER, MATRIX_NAME)

# Channel Models
ch_models = ['CDL-D', 'UMa']
rt_scens = ['asu_campus_3p5']
models = rt_scens + ch_models

#%% [SIONNA ENV] Load and Prepare Data

from data_gen import DataConfig, load_data_matrices
from model_config import ModelConfig

# Configure data generation
data_cfg = DataConfig(
    n_samples = 10_000,
    n_prbs = 20,
    n_rx = 1,
    n_tx = 32,
    snr = 50,
    normalize = 'dataset-mean-var-complex',
)

# UMAP parameters
data_cfg.x_points = data_cfg.n_samples #int(2e5)  # Number of points to sample from each dataset (randomly)
data_cfg.seed = 42  # Set to None to keep random
data_cfg.rt_uniform_steps = [3, 3] if data_cfg.n_samples <= 10_000 else [1, 1]

# Load data
data_matrices = load_data_matrices(models, data_cfg)

#%% [SIONNA ENV] Save matrices
os.makedirs(DATA_FOLDER, exist_ok=True)
with open(MAT_PATH, 'wb') as f:
    pickle.dump(data_matrices, f)

#%% [PYTORCH ENV] Load matrices

with open(MAT_PATH, 'rb') as f:
    data_matrices = pickle.load(f)

models = list(data_matrices.keys())

#%% [PYTORCH ENV] Create Base Configuration

from model_config import ModelConfig
from thtt_utils import train_models, cross_test_models
from thtt_plot import plot_training_results, plot_test_matrix

# Create base model configuration
base_config = ModelConfig(
    # Model architecture
    encoded_dim=64,   # 32x reduction (NC=16 * n_ant=64 * 2 IQ / 64 = 32x)
    n_refine_nets=6,  # != 1 for CSInetPlus | -1 for TransformerAE
    n_taps=16,
    n_antennas=32,  # TODO: change to 64
    
    # Training parameters
    train_batch_size=16,
    num_epochs=15,
    learning_rate=1e-2,
    
    # Directory structure
    dataset_main_folder='channel_experiment'
)

#%% Train Models

# Train models using base config
all_res = train_models(data_matrices, base_config)

# Plot training results
plot_training_results(all_res, models)

#%% Cross-Test Models

# Test models after initial training
print("\nCross-testing base models...")
all_test_results, results_matrix = cross_test_models(data_matrices, base_config, use_finetuned=False)

# Plot test matrix
plot_test_matrix(results_matrix, models)

# Print detailed results
results_matrix_db = 10 * np.log10(results_matrix)
df = pd.DataFrame(results_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nBase Model Test Results (NMSE in dB):")
print("==========================")
print(df.to_string())

#%% Cross-fine-tune

# Calculate sizes for train/test split
percent = 0.01  # Use 40% for fine-tuning
n_samples = data_matrices[models[0]].shape[0]
n_train = int(n_samples * percent)

# Create train/test splits for each model
train_data = {}
test_data = {}
np.random.seed(42)  # Use fixed seed for reproducibility

for model in data_matrices.keys():
    # Randomly shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split data
    train_data[model] = data_matrices[model][train_indices]
    test_data[model] = data_matrices[model][test_indices]

# Fine-tune each source->target pair
for source_idx, source_model in enumerate(models):
    for target_idx, target_model in enumerate(models):
        if target_model == source_model:
            continue
        
        title = f'Fine-tuning {source_model} -> {target_model}'
        print(f"\n{title}")
        
        # Create config for this specific fine-tuning pair
        pair_config = base_config.for_finetuning(
            source_model=source_model,
            num_epochs=5,
            n_train_samples=n_train
        )
        
        # Fine-tune model using training subset
        target_train_data = {target_model: train_data[target_model]}
        results = train_models(target_train_data, pair_config)

        plot_training_results(results, [source_model], title=title)

#%% Test all fine-tuned models on test data
print(f"\nCross-testing fine-tuned models on {percent*100}% of data...")

# Test using fine-tuned models
all_test_results, results_matrix = cross_test_models(test_data, base_config, use_finetuned=True)

# Plot fine-tuning test results
plot_test_matrix(results_matrix, models)

# Print detailed results
results_matrix_db = 10 * np.log10(results_matrix)
df = pd.DataFrame(results_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nFine-tuning Test Results (NMSE in dB):")
print("=====================================")
print(df.to_string())

# Future work: add a way to measure the performance DROP in the fine-tuned models
#              E.g. test the fine-tuned model always on the source model's data
