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
MATRIX_NAME = 'data_matrices_50k_complex_all.pkl'
MAT_PATH = os.path.join(DATA_FOLDER, MATRIX_NAME)

# Channel Models
ch_models = ['CDL-C', 'UMa'] # 'UMa!param!asu_campus_3p5'
rt_scens = ['asu_campus_3p5', 'city_0_newyork_3p5']
models = rt_scens + ch_models

NT = 32
NC = 16

#%% [SIONNA ENV] Load and Prepare Data

from data_gen import DataConfig, load_data_matrices

# Configure data generation
data_cfg = DataConfig(
    n_samples = 50_000,
    n_prbs = 20,
    n_rx = 1,
    n_tx = NT,
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
from thtt_plot import (plot_training_results, plot_test_matrix, 
                       plot_pretraining_comparison)

# Create base model configuration
base_config = ModelConfig(
    # Model architecture
    encoded_dim=64,   # 32x reduction (NC=16 * n_ant=64 * 2 IQ / 64 = 32x)
    n_refine_nets=6,  # != 1 for CSInetPlus | -1 for TransformerAE
    n_taps=NC,
    n_antennas=NT,
    
    # Training parameters
    train_batch_size=16,
    num_epochs=15,
    learning_rate=1e-2,
    
    # Directory structure
    dataset_main_folder='channel_experiment_all'
)

#%% [PYTORCH ENV] Train Models

# Train models using base config
all_res = train_models(data_matrices, base_config)

# Plot training results
plot_training_results(all_res, models)

#%% [PYTORCH ENV] Cross-Test Models

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

#%% [PYTORCH ENV] Cross-fine-tune

# Calculate sizes for train/test split
percent = 0.9  # Use 40% for fine-tuning
samp_per_model = [data_matrices[model].shape[0] for model in models]
n_samples = min(samp_per_model) # only generate indices up to this number
n_train = int(np.median(samp_per_model) * percent)

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

res_list = []
fine_tune_models = []
# Fine-tune each source->target pair
for source_idx, source_model in enumerate(models):
    for target_idx, target_model in enumerate(models):
        if target_model == source_model:
            continue
        
        title = f'Fine-tuning {source_model} -> {target_model}'
        print(f"\n{title}")
        
        # Fine saved models to "_finetuned" folder internally
        if os.path.exists(f'./channel_experiment_all_finetuned/model_{source_model}_{target_model}.pth'):
            print(f"Model {source_model} -> {target_model} already exists")
            continue
        
        # Create config for this specific fine-tuning pair
        pair_config = base_config.for_finetuning(
            source_model=source_model,
            num_epochs=15,
            n_train_samples=n_train
        )
        
        # Fine-tune model using training subset
        target_train_data = {target_model: train_data[target_model]}
        results = train_models(target_train_data, pair_config)

        plot_training_results(results, [source_model], title=title)

        res_list.append(results)
        fine_tune_models.append(title)

# plot_training_results(res_list, fine_tune_models)

#%% [PYTORCH ENV] Test all fine-tuned models on test data
print(f"\nCross-testing fine-tuned models on {percent*100}% of data...")

# Test using fine-tuned models
finetuned_test_results, finetuned_results_matrix = \
    cross_test_models(test_data, base_config, use_finetuned=True)

# Plot fine-tuning test results
plot_test_matrix(finetuned_results_matrix, models)

# Print detailed results
results_matrix_db = 10 * np.log10(finetuned_results_matrix)
df = pd.DataFrame(results_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nFine-tuning Test Results (NMSE in dB):")
print("=====================================")
print(df.to_string())

# Future work: add a way to measure the performance DROP in the fine-tuned models
#              E.g. test the fine-tuned model always on the source model's data

#%% [PYTORCH ENV] Evaluate GAIN from fine-tuning

gain_matrix = finetuned_results_matrix / results_matrix

# Plot fine-tuning test results
plot_test_matrix(gain_matrix, models)

# Print detailed results
gain_matrix_db = 10 * np.log10(gain_matrix)
df = pd.DataFrame(gain_matrix_db, index=models, columns=models)
df = df.round(1)

print("\nFine-tuning Test Results (NMSE Gain in dB):")
print("===========================================")
print(df.to_string())


#%% [PYTORCH ENV] Compare pre-trained vs non-pre-trained models

# Configuration
data_percents = [1, 5, 10, 20, 40, 60, 80]  # Percentages of training data to use
models = ['asu_campus_3p5', 'city_0_newyork_3p5', 'CDL-C', 'UMa']
base_model = models[0]  # Model to train from scratch
pretrained_models = models[1:]  # Models to use for pre-training

# Pre-train models (ONLY if not done before)
new_base_config = base_config.clone(dataset_main_folder='channel_experiment_all_percentages')
pre_train = False
if pre_train:
    data_matrices_to_pre_train = {model: data_matrices[model] for model in pretrained_models}
    all_res = train_models(data_matrices_to_pre_train, new_base_config)
    plot_training_results(all_res, pretrained_models, title='Pre-training')

# Calculate sizes for train/test split
samp_per_model = [data_matrices[model].shape[0] for model in models]
n_samples = min(samp_per_model)  # Use minimum samples across all models
n_train_total = int(n_samples * max(data_percents) / 100)

# Create fixed test set indices
np.random.seed(42)  # Use fixed seed for reproducibility
all_indices = np.random.permutation(n_samples)

# Prepare test data for all models
test_data = data_matrices[base_model][test_indices]

# Initialize results matrix: rows=data_percents, cols=[base_model + pretrained_models]
results_matrix = np.zeros((len(data_percents), len(models)))
results_matrix_db = np.zeros_like(results_matrix)

# For each data percentage
for perc_idx, data_percent in enumerate(data_percents):
    print(f"\nTraining with {data_percent}% of data...")
    
    # Calculate number of training samples for this percentage
    n_train = int(n_train_total * data_percent / 100)
    train_indices = all_indices[:n_train]
    test_indices = all_indices[n_train:]
    # max_test_size = int(max(data_percents)/100*n_samples)
    # test_indices = all_indices[max_test_size:]
    # Test size: keep same size throughout (set to min size), or adapt it

    # Prepare training data
    train_data = data_matrices[base_model][train_indices]
    # Step 1: Train base model from scratch
    print(f"\nTraining {base_model} from scratch...")

    # Train base model
    base_results = train_models({base_model: train_data}, new_base_config)
    
    # Test base model
    test_results, test_matrix = cross_test_models(
        {base_model: test_data}, 
        new_base_config,
        use_finetuned=False
    )
    results_matrix[perc_idx, 0] = test_matrix[0, 0]  # Store base model result
    
    # Step 2: Fine-tune each pre-trained model
    for model_idx, pretrained_model in enumerate(pretrained_models):
        print(f"\nFine-tuning {pretrained_model} model...")
        
        # Create config for fine-tuning (load from previous pre-training)
        finetune_config = new_base_config.for_finetuning(
            source_model=pretrained_model,
            num_epochs=15,  # Adjust as needed
        )
        
        # Fine-tune model
        finetune_results = train_models({base_model: train_data}, finetune_config)
        
        # Test fine-tuned model
        test_results, test_matrix = cross_test_models(
            {base_model: test_data}, 
            finetune_config,
            use_finetuned=True,
            load_source_from_config=True
        )
        results_matrix[perc_idx, model_idx + 1] = test_matrix[0, 0]
    
    # Convert results to dB for this percentage
    results_matrix_db[perc_idx] = 10 * np.log10(results_matrix[perc_idx])
    
    # Print results for this percentage
    print(f"\nResults for {data_percent}% training data (NMSE in dB):")
    print("=" * 50)
    print(f"Base model ({base_model}): {results_matrix_db[perc_idx, 0]:.1f} dB")
    for i, model in enumerate(pretrained_models):
        print(f"Pre-trained {model}: {results_matrix_db[perc_idx, i + 1]:.1f} dB")
    

# Print final results table
print("\nFinal Results (NMSE in dB):")
print("=" * 50)
df = pd.DataFrame(
    results_matrix_db,
    index=[f"{p}%" for p in data_percents],
    columns=[base_model] + pretrained_models
)
print(df.round(1).to_string())

# Save results matrix
os.makedirs('./results', exist_ok=True)
np.save('./results/pretraining_results.npy', results_matrix_db)

#%% Plot results for publication

# Plot performance comparison
plot_pretraining_comparison(
    data_percents=data_percents,
    results_matrix_db=results_matrix_db,
    models=[base_model] + pretrained_models,
    save_path='./results',
    plot_type='performance'
)

# Plot gain comparison
plot_pretraining_comparison(
    data_percents=data_percents,
    results_matrix_db=results_matrix_db,
    models=[base_model] + pretrained_models,
    save_path='./results',
    plot_type='gain'
)

# %%
