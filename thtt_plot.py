import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

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