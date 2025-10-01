#%% [SIONNA ENV] Append validation losses

import os
from pathlib import Path
import pandas as pd

def merge_validation_csvs(
    folder: Path,
    file1: str = "validation_losses.csv",
    file2: str = "validation_losses2.csv",
    out: str = "validation_losses-final.csv",
    suffix_from: str = "v2",
    safe: bool = True,
    overwrite: bool = False,
):
    f1 = folder / file1
    f2 = folder / file2
    fout = folder / out

    if not f1.exists() or not f2.exists():
        print(f"[SKIP] Missing files in {folder}: "
              f"{'missing validation-loss.csv' if not f1.exists() else ''} "
              f"{'and' if (not f1.exists() and not f2.exists()) else ''} "
              f"{'missing validation-loss2.csv' if not f2.exists() else ''}".strip())
        return

    # Read
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    
    # Drop Horizon explicitly from df2
    df2 = df2.drop(columns=[c for c in df2.columns if c == "horizon"])

    # Disambiguate overlapping column names by suffixing df2 columns that collide
    df2_renamed = df2.copy()
    new_cols = []
    for col in df2.columns:
        if col in df1.columns:
            new_cols.append(f"{col}_{suffix_from}")
        else:
            new_cols.append(col)
    df2_renamed.columns = new_cols

    # Merge by index (assumes same row order/length; adjust to a key join if needed)
    merged = pd.concat([df1, df2_renamed], axis=1)

    if safe:
        print(f"[DRY-RUN] Would write {fout}")
        print(f"  - From: {f1.name} (cols={list(df1.columns)})")
        print(f"  - Plus: {f2.name} (cols={list(df2.columns)}) "
              f"→ added as {list(df2_renamed.columns)}")
        print(f"  - Result shape: {merged.shape}")
        return

    if fout.exists() and not overwrite:
        print(f"[SKIP] {fout} already exists. Use overwrite=True to replace.")
        return

    merged.to_csv(fout, index=False)
    print(f"[OK] Wrote {fout} (shape={merged.shape})")


if __name__ == "__main__":
    # Parameters
    dopplers = [10, 100, 400]
    interps = [2, 10, 100]
    steps = 60  # from your example name
    base_dir = Path(".").resolve()  # change if needed

    # Toggle this to run for real
    SAFE_MODE = False   # dry-run
    OVERWRITE = True  # only used when SAFE_MODE == False

    for d in dopplers:
        for k in interps:
            folder_name = f"ch_pred_models_{d}hz_{steps}steps_INTERP_{k}"
            folder_path = base_dir / folder_name
            merge_validation_csvs(
                folder=folder_path,
                suffix_from="+Doppler",   # columns copied from validation-loss2.csv get "_v2"
                safe=SAFE_MODE,
                overwrite=OVERWRITE,
            )

#%% Plot validation losses

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_validation_losses(folder: Path, csv_name: str = "validation_losses-final.csv"):
    """Create a validation-loss plot from a merged CSV and save a PNG next to it."""
    csv_path = folder / csv_name
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Locate horizon column (case-insensitive)
    horizon_col = next((c for c in df.columns if c.lower() == "horizon"), None)
    if horizon_col is None:
        print(f"[SKIP] No horizon column in {csv_path}")
        return
    horizons = df[horizon_col].tolist()

    # Detect base model names that have both *_gru_best and *_sh
    suffix_best = "_gru_best"
    suffix_sh   = "_sh"
    cols = set(df.columns)
    models = [
        c[:-len(suffix_best)]
        for c in df.columns if c.endswith(suffix_best)
        if f"{c[:-len(suffix_best)]}{suffix_sh}" in cols
    ]
    if not models:
        print(f"[SKIP] No model columns in {csv_path}")
        return

    colors  = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628']
    markers = ['o', 's', 'D', 'P', '^', 'v']

    plt.figure(dpi=300)
    for i, m in enumerate(models):
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(horizons, df[f"{m}{suffix_best}"], label=f"{m}_best",
                 color=color, marker=marker, markersize=5)
        plt.plot(horizons, df[f"{m}{suffix_sh}"], label=f"{m}_SH",
                 color=color, linestyle="--", marker=marker, markersize=5)

    plt.xlabel("Horizon (ms)")
    plt.ylabel("Validation Loss (NMSE in dB)")
    plt.legend(ncols=min(4, len(models)), bbox_to_anchor=(0.5, 1.02), loc="lower center")
    plt.xlim(0, max(horizons) + 0.5)
    plt.grid(True)
    plt.tight_layout()

    out_path = folder / (csv_path.stem + ".png")
    plt.show()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")

# ---- loop over all folders ----
dopplers = [10, 100, 400]
interps  = [2, 10, 100]
steps    = 60
base_dir = Path(".").resolve() / "ch_pred_results"

for d in dopplers:
    for k in interps:
        folder = base_dir / f"ch_pred_models_{d}hz_{steps}steps_INTERP_{k}"
        plot_validation_losses(folder)

#%% Plot validation losses grid

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
base_dir = Path(".").resolve() / "ch_pred_results"
steps    = 60
dopplers = [10, 100, 400]          # left -> right
interps  = [100, 10, 2]            # top -> bottom (descending as requested)
csv_name = "validation_losses-final.csv"
out_name = "validation_losses-grid.png"

# Color/marker per normalized model label to keep consistent across subplots
MODEL_STYLES = {
    "TDL": {"color": "#E41A1C", "marker": "o"},
    "CDL": {"color": "#377EB8", "marker": "s"},
    "UMA": {"color": "#4DAF4A", "marker": "D"},
    "ASU": {"color": "#984EA3", "marker": "P"},
}
BEST_SUFFIX = "_gru_best"   # solid line
SH_SUFFIX   = "_sh"         # dashed line
# ----------------------------

def simplify_model_name(raw_base: str) -> str:
    """Map raw base column prefix to one of: TDL, CDL, UMA, ASU (fallback: uppercased raw)."""
    b = raw_base.lower()
    if "asu" in b or "campus" in b:
        return "ASU"
    if "tdl" in b:
        return "TDL"
    if "cdl" in b:
        return "CDL"
    if "uma" in b or "urbanmacro" in b or "urban_macro" in b:
        return "UMA"
    return raw_base.upper()

def detect_models(df: pd.DataFrame) -> list[str]:
    """Return normalized model labels present in df that have BOTH *_gru_best and *_sh."""
    cols = set(df.columns)
    found = []
    for c in df.columns:
        if c.endswith(BEST_SUFFIX):
            base = c[:-len(BEST_SUFFIX)]
            if f"{base}{SH_SUFFIX}" in cols:
                label = simplify_model_name(base)
                if label not in found:
                    found.append(label)
    # Keep only those we style
    return [m for m in found if m in MODEL_STYLES]

def get_horizon_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if lc == "horizon" or lc == "horizons":
            return c
    return None

def plot_folder_on_ax(folder: Path, ax, models_in_use: list[str] | None = None):
    """Plot a single folder's CSV on the provided axes (no legend)."""
    csv_path = folder / csv_name
    if not csv_path.exists():
        ax.set_visible(False)
        return None

    df = pd.read_csv(csv_path)
    hcol = get_horizon_col(df)
    if hcol is None:
        ax.set_visible(False)
        return None

    # If models set not provided, detect from this df
    models = models_in_use or detect_models(df)
    if not models:
        ax.set_visible(False)
        return None

    horizons = df[hcol].tolist()

    # Plot each model's two lines
    for m in models:
        style = MODEL_STYLES[m]
        # Find raw base(s) that map to this simplified label (handle multiple matches robustly)
        # Prefer exact one; if multiple match, plot all under same style (rare).
        bases = []
        for c in df.columns:
            if c.endswith(BEST_SUFFIX):
                base = c[:-len(BEST_SUFFIX)]
                if simplify_model_name(base) == m and f"{base}{SH_SUFFIX}" in df.columns:
                    bases.append(base)
        for base in bases:
            ax.plot(horizons, df[f"{base}{BEST_SUFFIX}"],
                    label=f"{m}", color=style["color"], marker=style["marker"], markersize=4, linewidth=1.2)
            ax.plot(horizons, df[f"{base}{SH_SUFFIX}"],
                    label=f"{m} SH", color=style["color"], linestyle="--", marker=style["marker"], markersize=3, linewidth=1.0)

    ax.grid(True)
    try:
        ax.set_xlim(0, max(horizons) + 0.5)
    except Exception:
        pass
    return models

# ---------- build grid ----------
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9), dpi=500, sharex=True, sharey=True)

# First pass: determine a unified model set (intersection across all reachable CSVs)
unified_models = None
for r, k in enumerate(interps):
    for c, d in enumerate(dopplers):
        folder = base_dir / f"ch_pred_models_{d}hz_{steps}steps_INTERP_{k}"
        csv_path = folder / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        hcol = get_horizon_col(df)
        if hcol is None:
            continue
        ms = set(detect_models(df))
        if not ms:
            continue
        unified_models = set(ms) if unified_models is None else unified_models & ms

unified_models = [m for m in MODEL_STYLES if m in unified_models]

# Second pass: plot each cell using the unified model set
handles_for_legend = {}
for r, k in enumerate(interps):
    for c, d in enumerate(dopplers):
        ax = axes[r, c]
        folder = base_dir / f"ch_pred_models_{d}hz_{steps}steps_INTERP_{k}"
        present = plot_folder_on_ax(folder, ax, models_in_use=unified_models)

        # Titles on top row
        if r == 0:
            ax.set_title(f"Doppler {d} Hz", fontsize=10)
        # Row labels on first column
        if c == 0:
            ax.set_ylabel(f"Interp {k}", fontsize=10)
        # Axis labels on edges only (shared axes)
        if r == len(interps) - 1:
            ax.set_xlabel("Horizon (ms)")
        if c == 0:
            ax.set_ylabel(f"Interp {k}\nValidation Loss (NMSE dB)")

# Build a single, uniform legend (order: TDL, CDL, UMA, ASU if present)
legend_labels = []
legend_lines  = []
for m in unified_models:
    style = MODEL_STYLES[m]
    # create proxy artists for legend (solid: best, dashed: SH)
    solid, = plt.plot([], [], linestyle="-",  color=style["color"], marker=style["marker"], label=m)
    dash,  = plt.plot([], [], linestyle="--", color=style["color"], marker=style["marker"], label=f"{m} SH")
    legend_lines.extend([solid, dash])
    legend_labels.extend([m, f"{m} SH"])

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.suptitle("Validation Loss Across Doppler (Columns) and Interp (Rows)", y=1.01, fontsize=12)

# One global legend on top
fig.legend(legend_lines, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.50, 0.99))

plt.show()  # show before saving
fig.savefig(base_dir / out_name, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved {base_dir / out_name}")

#%% COMPUTE CROSS-TEST NMSE MATRIX

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from thtt_ch_pred_utils import compute_nmse_matrix

# ===== USER CONFIG =====
base_dir  = Path(".").resolve()
steps     = 60
N_SAMPLES = 200_000
dopplers  = [10, 100, 400]   # columns (left→right)
interps   = [100, 10, 2]     # rows (top→bottom)
models_base = ['TDL-A', 'CDL-C', 'UMa']   # ASU handled separately
HORIZON   = 1
cache_dir = base_dir / "nmse_cache"       # where matrices are stored
grid_png  = base_dir / "nmse-confusion-grid.png"
# =======================

# ---- ASU helpers ----
def asu_model_name(interp_factor: int) -> str:
    """Exact model folder name for loading (used by compute)."""
    return f"asu_campus_3p5_10cm_interp_{interp_factor}"

def asu_label(interp_factor: int) -> str:
    """Pretty label for plotting (used by plot)."""
    return "ASU-40mm" if interp_factor == 2 else f"ASU-{100 // interp_factor}mm"

# ---- CACHING LAYOUT ----
def cell_cache_path(interp: int, dop: int) -> Path:
    """Standard filename for a (row=interp, col=doppler) cell."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"nmse_matrix_interp{interp}_dop{dop}.npz"

# ================== STAGE 1: COMPUTE & CACHE ==================
def compute_and_cache_confusion_grid(L_IN, NT, overwrite: bool = False) -> None:
    """
    Compute the NMSE matrices for all (interp, doppler) cells and save to cache_dir as .npz.
    No plotting happens here.
    """
    for interp in interps:
        for dop in dopplers:
            out_path = cell_cache_path(interp, dop)
            if out_path.exists() and not overwrite:
                print(f"[SKIP] {out_path.name} exists (use overwrite=True to redo).")
                continue

            models_folder = base_dir / f"ch_pred_models_{dop}hz_{steps}steps_INTERP_{interp}"
            data_folder   = base_dir / f"ch_pred_data_{N_SAMPLES//1000}k_{dop}hz_{steps}steps"

            models_for_compute = [*models_base, asu_model_name(interp)]
            display_labels     = [*models_base, asu_label(interp)]

            # ---- YOUR compute function (must be defined/imported) ----
            results_matrix = compute_nmse_matrix(
                models_for_compute,
                horizon=HORIZON,
                l_in=L_IN,
                models_folder=models_folder,
                data_folder=data_folder,
                num_tx_antennas=NT
            )

            # Persist everything needed to plot later
            np.savez_compressed(
                out_path,
                results_matrix=results_matrix,
                display_labels=np.array(display_labels, dtype=object),
                models_for_compute=np.array(models_for_compute, dtype=object),
                interp=np.int32(interp),
                doppler=np.int32(dop),
                models_folder=str(models_folder),
                data_folder=str(data_folder),
            )
            print(f"[OK] Cached {out_path.name}  shape={results_matrix.shape}")

compute_and_cache_confusion_grid(L_IN=steps, NT=NT, overwrite=False)

#%% Plot cross-test NMSE matrix

# ================== STAGE 2: LOAD & PLOT ==================
def load_cell(interp: int, dop: int):
    p = cell_cache_path(interp, dop)
    if not p.exists():
        raise FileNotFoundError(f"Missing cache file: {p}")
    z = np.load(p, allow_pickle=True)
    return {
        "M": z["results_matrix"],
        "labels": list(z["display_labels"]),
        "interp": int(z["interp"]),
        "doppler": int(z["doppler"]),
    }

# --- Much better grid plot from cache (uniform colorbar, clean labels) ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

# assumes: interps, dopplers, base_dir, grid_png, load_cell are defined as in your cache code

def plot_confusion_grid_from_cache(
    *,
    figsize=(12, 8),
    dpi=300,
    cmap="viridis_r",
    annotate=True,
    vmin_db: Optional[float] = None,
    vmax_db: Optional[float] = None,
    suptitle="Cross-Test NMSE (dB) — Columns: Doppler, Rows: Interp (from cache)"
) -> None:
    """Render the 3x3 grid from cached .npz files with tight spacing,
    y-axis labels on the first column, and one shared colorbar."""
    # 1) Load all cells
    cells = {}
    all_vals = []
    for r, interp in enumerate(interps):
        for c, dop in enumerate(dopplers):
            try:
                cell = load_cell(interp, dop)  # {"M":..., "labels":..., ...}
                cells[(r, c)] = cell
                all_vals.append(10 * np.log10(cell["M"]))
            except FileNotFoundError:
                cells[(r, c)] = None

    # 2) Global color limits
    if not all_vals:
        raise RuntimeError("No cached matrices found.")
    stacked = np.concatenate([x.ravel() for x in all_vals])
    if vmin_db is None: vmin_db = float(np.nanmin(stacked))
    if vmax_db is None: vmax_db = float(np.nanmax(stacked))

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=figsize, dpi=dpi,
        sharex=True, sharey=True,
        gridspec_kw={"wspace": -0.5, "hspace": 0.05}  # tight spacing
    )

    for r, interp in enumerate(interps):
        for c, dop in enumerate(dopplers):
            ax = axes[r, c]
            cell = cells[(r, c)]
            if cell is None:
                ax.axis("off")
                continue

            labels = cell["labels"]

            # Column titles
            if r == 0:
                ax.set_title(f"Doppler {dop} Hz", fontsize=11, pad=4)
            # Row labels on first column
            if c == 0:
                ax.set_ylabel(f"Interp {interp}\nTraining Model", fontsize=11, labelpad=6)

            # Plot without per-axes colorbar
            plot_test_matrix(
                cell["M"], labels, ax=ax,
                annotate=annotate,
                cmap=cmap,
                vmin_db=vmin_db,
                vmax_db=vmax_db,
                tick_font=8, text_font=8, label_font=9,
                title=None
            )

            # Hide inner tick labels except bottom row / first column
            if r < len(interps) - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    # Single shared colorbar
    norm = plt.Normalize(vmin=vmin_db, vmax=vmax_db)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.04, pad=0.02)
    cbar.ax.set_ylabel("NMSE (dB)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(suptitle, x=0.62, y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))  # tighten after adding suptitle

    plt.show()
    fig.savefig(grid_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {grid_png}")

plot_confusion_grid_from_cache(annotate=True, vmin_db=-30, vmax_db=0)

#%% Final plots

from thtt_plot import plot_validation_losses_from_csv

folder = base_dir / "FINAL_ch_pred_models_100hz_60steps_INTERP_10"
csv_path = folder / "validation_losses-final2.csv"

out_path = folder / "validation_losses.png"
plot_validation_losses_from_csv(csv_path, out_path)

#%%

# plot_test_matrix()