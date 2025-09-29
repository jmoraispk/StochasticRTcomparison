
#%%
import pandas as pd

import matplotlib.pyplot as plt

doppler = 0
interp = 5

path = '/home/joao/Documents/GitHub/StochasticRTcomparison/ch_pred_results/new_rt_results'
folder = path + f'/new_rt_ch_pred_models_{doppler}hz_60steps_INTERP_{interp}'


# read validation .csv
df = pd.read_csv(folder + '/validation_losses3.csv')

def plot_validation_loss(df, horizons, ax=None, lims=None):

    build_ax = ax is None

    if build_ax:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

    for col in df.columns:
        if col == 'horizon':
            continue
        ax.plot(horizons, df[col], marker='o', label=col)

    ax.set_xlabel("Horizon (ms)")
    ax.set_ylabel("Validation Loss (NMSE dB)")
    ax.set_title("Validation Loss vs Horizon")
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    if build_ax:
        ax.legend()
    if lims is not None:
        ax.set_ylim(lims)

plot_validation_loss(df, df['horizon'].values)

#%%

dopplers = [0, 10, 30, 100, 200]
interps = [5, 10, 30]

fig, axs = plt.subplots(len(dopplers), len(interps), figsize=(10, 12), dpi=300)

for i, doppler in enumerate(dopplers):
    for j, interp in enumerate(interps):
        folder = path + f'/new_rt_ch_pred_models_{doppler}hz_60steps_INTERP_{interp}'
        df = pd.read_csv(folder + '/validation_losses3.csv')
        horizons = df['horizon'].values
        plot_validation_loss(df, horizons, ax=axs[i, j], lims=(-30, 5))
    # break

plt.tight_layout()
plt.show()

#%%

"""
- Likely Doppler Resonance:
Doppler phase shifts will be similar to a channel prediction
a few steps ahead, leading to a better performance in that region.

- The model struggles to predict channels that change very slowly.
(100 interpolation collapses.)




"""