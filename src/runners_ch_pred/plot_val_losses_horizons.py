
#%%

from thtt_ch_pred_plot import plot_validation_losses_from_csv
from pathlib import Path

base_dir = Path(".").resolve() / "ch_pred_results"
folder = base_dir / "FINAL_ch_pred_models_100hz_60steps_INTERP_10"
csv_path = folder / "validation_losses-final.csv"

out_path = folder / "validation_losses.png"
plot_validation_losses_from_csv(csv_path, out_path, split_legend=True)
