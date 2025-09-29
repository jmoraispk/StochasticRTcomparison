import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import os
import matplotlib as mpl
from matplotlib.colors import Normalize

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

def plot_test_matrix(
    results_matrix: np.ndarray,
    models: list[str],
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    *,
    annotate: bool = True,
    cmap: str = "viridis_r",
    vmin_db: Optional[float] = None,
    vmax_db: Optional[float] = None,
    tick_font: int = 9,
    text_font: int = 10,
    label_font: int = 11,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot NMSE matrix (in dB) as a heatmap. If ax is provided, draw there and DO NOT add a colorbar.
    Returns the axes used (and you can grab its 'images[-1]' as the mappable).
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        own_fig = True

    M_db = 10 * np.log10(results_matrix)
    if vmin_db is None:
        vmin_db = float(np.nanmin(M_db))
    if vmax_db is None:
        vmax_db = float(np.nanmax(M_db))

    im = ax.imshow(M_db, cmap=cmap, norm=Normalize(vmin=vmin_db, vmax=vmax_db))

    # Annotations (optional, smaller font)
    if annotate:
        mean_val = float(np.nanmean(M_db))
        for i in range(len(models)):
            for j in range(len(models)):
                val = M_db[i, j]
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center",
                    color="white" if val > mean_val else "black",
                    fontsize=text_font,
                )

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=tick_font)
    ax.set_yticklabels(models, fontsize=tick_font)
    if title:
        ax.set_title(title, fontsize=label_font)

    # Only save/show when we own the figure (standalone usage)
    if own_fig:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("NMSE (dB)", fontsize=label_font)
        cbar.ax.tick_params(labelsize=tick_font)
        ax.set_ylabel("Training Model", fontsize=label_font)
        ax.set_xlabel("Testing Model", fontsize=label_font)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    return ax


def plot_pretraining_comparison(x_values: list,
                              results_matrix_db: np.ndarray,
                              models: list,
                              save_path: str = './results',
                              plot_type: str = 'performance',
                              x_label: str = 'Training Data (%)',
                              x_logscale: bool = False,
                              legend_labels: Optional[list] = None,
                              legend_order: Optional[list[int]] = None) -> None:
    """Plot pre-training comparison results.
    
    Args:
        x_values: List of x-axis values (can be percentages, number of datapoints, etc.)
        results_matrix_db: Matrix of results in dB where rows=x_values, cols=models
        models: List of model names
        save_path: Path to save the plots
        plot_type: Either 'performance' or 'gain' to choose plot type
        x_label: Label for x-axis. If it contains '%', values will be formatted as percentages,
                otherwise large numbers will be formatted with commas
        x_logscale: If True, use logarithmic scale for the x-axis.
        legend_labels: Optional list of custom legend labels (must match number of plotted series).
        legend_order: Optional list of indices specifying legend entry order (0-based).
    """
    # Set up publication-quality plot settings
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14       # x-tick label font size
    mpl.rcParams['ytick.labelsize'] = 14       # y-tick label font size
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 12
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 1
    mpl.rcParams['legend.edgecolor'] = 'grey'
    mpl.rcParams['legend.title_fontsize'] = 15
    mpl.rcParams['axes.labelpad'] = 10

    # Define consistent markers and colors
    markers = ['o', 's', '^', 'D']  # Different marker for each line
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']  # Orange, Blue, Green, Red

    # Create figure
    _, ax = plt.subplots()
    
    if x_logscale:
        ax.set_xscale('log')

    # Plot each model's performance
    k = 0 if plot_type == 'performance' else 1
    gains = results_matrix_db[:, 1:] - results_matrix_db[:, :1]
    y = results_matrix_db if plot_type == 'performance' else gains

    expected_len = len(models) - k
    if legend_labels is not None and len(legend_labels) == expected_len:
        labels = legend_labels
    else:
        labels = [m.replace('_', ' ') for m in models[k:]]

    for i, model in enumerate(models[k:]):
        ax.plot(x_values, y[:, i], 
                marker=markers[i+k], label=labels[i],
                color=colors[i+k],
                markerfacecolor='white', markeredgewidth=2)

    if plot_type == 'performance':
        ax.set_ylabel('NMSE (dB)')
        # ax.set_title('Model Performance vs Training Data Size')
        legend_loc = 'upper right'
        filename = 'pretrain_comparison'

    else:  # gain plot
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('NMSE Gain over Base Model (dB)')
        # ax.set_title('Performance Gain from Pre-training')
        legend_loc = 'best'
        filename = 'pretrain_gain_comparison'

    # Legend handling with optional reordering
    handles, leg_labels = ax.get_legend_handles_labels()
    if legend_order is not None:
        # Keep only valid indices and preserve order provided
        order = [idx for idx in legend_order if 0 <= idx < len(handles)]
        if len(order) > 0:
            handles = [handles[i] for i in order]
            leg_labels = [leg_labels[i] for i in order]
    ax.legend(handles, leg_labels, title='Model', loc=legend_loc)

    # Common plot settings
    ax.set_xlabel(x_label)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_values)
    # Infer format from x_label
    if '%' in x_label:
        ax.set_xticklabels([f'{x}%' for x in x_values])
    else:
        ax.set_xticklabels([f'{x:,}' for x in x_values])

    # Add minor gridlines
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{filename}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}/{filename}.png', bbox_inches='tight', dpi=300)

    return ax
