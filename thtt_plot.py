import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import os
import matplotlib as mpl

def plot_training_results(all_res: list, models: list, title: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot training and validation NMSE for each area.
    
    Args:
        all_res: List of training results
        models: List of model names
        save_path: Optional path to save the plot
    """
    def to_db(x):
        return 10 * np.log10(x)

    plt.figure(figsize=(10, 6), dpi=200)

    # Define colors for each model
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for model_idx, model in enumerate(models):
        result = all_res[model_idx]
        train_nmse_db = to_db(np.array(result['all_train_nmse']))
        val_nmse_db = to_db(np.array(result['all_val_nmse']))
        epochs = np.arange(1, len(train_nmse_db) + 1)
        
        # Use same color for both train and val lines
        plt.plot(epochs, train_nmse_db, '-', label=f'{model}', 
                 linewidth=2, color=colors[model_idx])
        plt.plot(epochs, val_nmse_db, '--', linewidth=2, color=colors[model_idx])
        
        test_nmse_db = to_db(result['test_nmse'])
        print(f'{model} - Final Test NMSE: {test_nmse_db:.2f} dB')

    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('NMSE (dB)', fontsize=16)
    plt.title(title if title else 'Training and Validation NMSE per Model', fontsize=14)
    plt.legend(fontsize=14)#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_test_matrix(results_matrix: np.ndarray, models: list, save_path: Optional[str] = None) -> None:
    """
    Plot the test results matrix as a heatmap.
    
    Args:
        results_matrix: Matrix of test results
        models: List of model names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8), dpi=200)
    
    results_matrix_db = 10 * np.log10(results_matrix)
    
    # Create heatmap
    plt.imshow(results_matrix_db, cmap='viridis_r')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('NMSE (dB)', fontsize=14)
    
    # Add text annotations
    # (first position of results is the training model, second is the testing model)
    for i in range(len(models)):
        for j in range(len(models)):
            plt.text(j, i, f'{results_matrix_db[i, j]:.1f}',
                    ha='center', va='center', color='white' 
                    if results_matrix_db[i, j] > np.mean(results_matrix_db) else 'black',
                    fontsize=16)
    
    plt.xticks(np.arange(len(models)), models, rotation=45, fontsize=14)
    plt.yticks(np.arange(len(models)), models, fontsize=14)
    plt.title('Cross-Test Results (NMSE in dB)', fontsize=14)
    plt.tight_layout()
    plt.ylabel('Training Model', fontsize=16)
    plt.xlabel('Testing Model', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_pretraining_comparison(data_percents: list,
                              results_matrix_db: np.ndarray,
                              models: list,
                              save_path: str = './results',
                              plot_type: str = 'performance') -> None:
    """Plot pre-training comparison results.
    
    Args:
        data_percents: List of training data percentages
        results_matrix_db: Matrix of results in dB where rows=percentages, cols=models
        models: List of model names
        save_path: Path to save the plots
        plot_type: Either 'performance' or 'gain' to choose plot type
    """
    # Set up publication-quality plot settings
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.8
    mpl.rcParams['legend.edgecolor'] = '0.8'

    # Define consistent markers and colors
    markers = ['o', 's', '^', 'D']  # Different marker for each line
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']  # Orange, Blue, Green, Red

    # Create figure
    _, ax = plt.subplots()


    # Plot each model's performance
    k = 0 if plot_type == 'performance' else 1
    gains = results_matrix_db[:, 1:] - results_matrix_db[:, :1]
    y = results_matrix_db if plot_type == 'performance' else gains
    for i, model in enumerate(models[k:]):
        ax.plot(data_percents, y[:, i], 
                marker=markers[i+k], label=model.replace('_', ' '),
                color=colors[i+k],
                markerfacecolor='white', markeredgewidth=2)

    if plot_type == 'performance':
        ax.set_ylabel('NMSE (dB)')
        ax.set_title('Model Performance vs Training Data Size')
        ax.legend(title='Models', loc='upper right')
        
        filename = 'pretrain_comparison'

    else:  # gain plot
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        ax.set_ylabel('NMSE Gain over Base Model (dB)')
        ax.set_title('Performance Gain from Pre-training')
        ax.legend(title='Pre-trained Models', loc='lower right')
        
        filename = 'pretrain_gain_comparison'

    # Common plot settings
    ax.set_xlabel('Training Data (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to show percentages
    ax.set_xticks(data_percents)
    ax.set_xticklabels([f'{p}%' for p in data_percents])

    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{filename}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}/{filename}.png', bbox_inches='tight', dpi=300)

    plt.show()