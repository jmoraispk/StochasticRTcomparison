# Stochastic RT Comparison

This project compares stochastic channel models with ray tracing models using UMAP visualization and machine learning techniques. It includes data generation (Sionna and DeepMIMO), topology analysis (UMAP via cuML), channel compression with autoencoders, and temporal channel prediction models.

## Install Dependencies

```
### ENV1: sionna environment (for data generation and topology under TensorFlow/Sionna)
mamba create -n env1_sionna python=3.11
mamba activate env1_sionna
pip install -r env1_sionna_requirements.txt

# Install cuML necessary for FAST UMAP transforms (check docs.rapids.ai/install)
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.4.*"

### ENV2: pytorch environment (for training models and experiments)
mamba create -n env2_pytorch python=3.11
mamba activate env2_pytorch

# Install PyTorch to run the models (check pytorch.org/get-started/locally)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -r env2_pytorch env2_pytorch_requirements.txt
uv pip install -e .   ( in deepmimo env)
uv pip install jupyter ipykernel

############# Install DeepMIMO command (in both env requirements)
# pip install deepmimo==4.0.0a3

# TIP: install uv (with `pip install uv`) and use `uv pip` instead of `pip` for speed
```

## Project Structure

```
├── data_gen.py                 # Core data generation and preparation
│   ├── DataConfig              # Configuration class for data generation
│   ├── load_data_matrices()    # Load/generate data for different models
│   ├── prepare_umap_data()     # Prepare data for UMAP analysis
│   └── outlier detection       # Functions for detecting and removing outliers
│
├── sionna_ch_gen.py            # Sionna channel generator wrapper
│
├── topology.py                 # UMAP visualization and analysis
│   ├── Data loading            # Load and prepare data using DataConfig
│   ├── UMAP computation        # Compute UMAP embeddings
│   ├── Visualization           # Plot embeddings with/without outliers
│   └── Partial UMAP fitting    # Fit UMAP on one model, transform another
│
├── topology_utils.py           # Visualization utilities
│   └── plot_umap_embeddings()  # Function for plotting UMAP embeddings
│
├── thtt.py                     # Training and testing (compression)
│   ├── Data preparation        # Prepare data for training
│   ├── Model training          # Train models on different datasets
│   └── Cross-testing           # Test models across different datasets
│
├── thtt_utils.py               # Training utilities
│   ├── convert_channel_angle_delay()  # Convert channel data format
│   └── train_val_test_split()         # Split data for training
│
├── thtt_plot.py                # Plotting utilities for compression experiments
│
├── csinet_train_test.py        # PyTorch training/testing for compression models
│
├── data_feed.py                # PyTorch Dataset for channel matrices
│
├── CsinetPlus.py               # CSI-Net+ baseline model (compression)
│
├── src/
│   └── thtt/
│       ├── compression/
│       │   └── transformerAE.py     # Transformer-based autoencoder
│       └── prediction/
│           └── transformerPred.py   # Transformer for channel prediction
│
├── thtt_ch_pred.py             # Temporal channel prediction experiment (GRU baseline)
│
├── thtt_ch_pred_utils.py       # Utilities for temporal prediction and NMSE matrices
│
├── thtt_ch_pred_plot.py        # Plotting for prediction experiments
│
├── nr_channel_predictor.py     # GRU-based temporal channel predictor
│
├── nr_channel_predictor_wrapper.py  # Train/predict/save/load helpers for GRU predictor
│
├── thtt_ch_pred2.py            # Runner for servers (temporal prediction)
│
├── append_val_losses.py        # Process compression results (thtt) and generate plots
│
├── rt_data_gen_loop.py         # Generate RT data for channel prediction
│
├── ch_compression_results/     # Saved results for compression experiments
│
├── ch_pred_results/            # Saved results for prediction experiments
│
├── nmse_cache/                 # Cached NMSE matrices
│
├── deepmimo_scenarios/         # DeepMIMO scenario metadata (if used)
│
├── slides/                     # Presentation materials
│
└── README.md
```

## Data Flow

```
data_gen.py ──────┐
                  ├─> topology.py (UMAP visualization)
                  │      └─> topology_utils.py (plotting)
                  │
                  └─> thtt.py (Training and testing)
                         └─> thtt_utils.py (data preparation)
```

## Key Modules and How They Interact

- data_gen.py
  - Defines `DataConfig` (generation parameters) and functions like `load_data_matrices()`.
  - Uses `SionnaChannelGenerator` from `sionna_ch_gen.py` and DeepMIMO to produce channel matrices.

- sionna_ch_gen.py
  - Provides `SionnaChannelGenerator` and `TopologyConfig` used by `data_gen.py`.

- thtt.py
  - Orchestrates the end-to-end flow: loads pickled matrices, constructs `ModelConfig`, calls `train_models()` and `cross_test_models()` from `thtt_utils.py`, and plots results.

- model_config.py
  - `ModelConfig` dataclass centralizes training hyperparameters, dataset directories, and provides `get_model_path()` helpers for base and fine-tuning scenarios.

- thtt_utils.py
  - Converts channel matrices to angle-delay domain (`convert_channel_angle_delay()`), performs splits, builds DataLoaders, and calls `train_model()` from `csinet_train_test.py`.

- csinet_train_test.py
  - Core PyTorch training: creates DataLoaders using `data_feed.DataFeed`, computes NMSE, trains either `CsinetPlus` or `TransformerAE` depending on config.
  - Imports models:
    - `from CsinetPlus import CsinetPlus`
    - `from transformerAE import TransformerAE` (update import path if using `src/thtt/compression/transformerAE.py`).

- data_feed.py
  - Implements `DataFeed` PyTorch Dataset. Converts complex channel matrices to real/imag `(2, Nc, Nt)` format expected by the models.

- CsinetPlus.py
  - CSI-Net+ style autoencoder baseline for channel compression.

- src/thtt/compression/transformerAE.py
  - Implements a transformer-based autoencoder for channel compression.
  - Classes:
    - `Encoder`: conv + TransformerEncoder over subcarriers, linear projection to `encoded_dim`.
    - `Decoder`: linear + TransformerEncoder + conv to reconstruct `(2, Nc, Nt)`.
    - `QuantizeLayer`: optional differentiable k-bit quantization.
    - `TransformerAE`: wrapper composing `Encoder` and `Decoder` with optional quantization.

- src/thtt/prediction/transformerPred.py
  - Implements a transformer encoder for temporal channel prediction with sequence input `(batch, seq_len, input_dim)` and final projection to `(batch, input_dim)`.
  - Class:
    - `TransformerModel`: input projection → TransformerEncoder → flatten → linear head.

- nr_channel_predictor.py and nr_channel_predictor_wrapper.py
  - GRU-based temporal channel predictor and training helpers (construct/train/predict/save/load).

- thtt_ch_pred.py and thtt_ch_pred_utils.py
  - End-to-end temporal prediction experiment logic: generate/process temporal channel `H`, split sequences, train GRU predictor, compute NMSE matrices across horizons, plot results.

- topology/topology.py and topology/topology_utils.py
  - UMAP embedding computation (optionally with cuML) and visualization helpers.

- append_val_losses.py, thtt_plot.py, thtt_ch_pred_plot.py
  - Plotting/aggregation utilities for experiments.

## Call/Usage Map (Selected Paths)

- thtt.py → thtt_utils.py → csinet_train_test.py → {CsinetPlus, TransformerAE} → data_feed.py
- thtt.py → thtt_plot.py (visualization)
- data_gen.py → sionna_ch_gen.py → DeepMIMO/Sionna backends
- thtt_ch_pred.py → nr_channel_predictor_wrapper.py → nr_channel_predictor.py → thtt_ch_pred_utils.py
- topology/topology.py → topology/topology_utils.py


## Usage

1. Configure data generation parameters in `data_gen.py` (`DataConfig`)
2. Generate/load data using `load_data_matrices()`
3. Visualize data topology using `topology/topology.py`:
   - Compute UMAP embeddings
   - Visualize with/without outliers
   - Perform partial UMAP fitting
4. Train and test compression models via `thtt.py`:
   - Data prep through `thtt_utils.py`
   - Train `CsinetPlus` or `TransformerAE`
   - Cross-test across datasets
5. Run temporal prediction via `thtt_ch_pred.py`:
   - Build sequences with `thtt_ch_pred_utils.py`
   - Train GRU predictor (`nr_channel_predictor.py`) or experiment with `transformerPred.py`

## Dependencies

- numpy, scipy
- matplotlib, seaborn
- cuML (for UMAP acceleration - only in topology)
- deepmimo (for ray tracing data)
- sionna (for stochastic channel models)
- torch, torchvision, tqdm, einops

## Notes

- GPU acceleration is recommended for both cuML UMAP and PyTorch models.
- Ensure your CUDA version matches the wheels you install for torch and cuML. (acheived through python dependencies)