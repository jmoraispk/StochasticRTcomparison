# Stochastic RT Comparison

This project compares stochastic channel models with ray tracing models using UMAP visualization and machine learning techniques.

## Project Structure

```
├── data_gen.py                 # Core data generation and preparation
│   ├── DataConfig              # Configuration class for data generation
│   ├── load_data_matrices()    # Load/generate data for different models
│   ├── prepare_umap_data()     # Prepare data for UMAP analysis
│   └── outlier detection       # Functions for detecting and removing outliers
│
├── topology.py                 # UMAP visualization and analysis
│   ├── Data loading           # Load and prepare data using DataConfig
│   ├── UMAP computation       # Compute UMAP embeddings
│   ├── Visualization          # Plot embeddings with/without outliers
│   └── Partial UMAP fitting   # Fit UMAP on one model, transform another
│
├── topology_utils.py          # Visualization utilities
│   └── plot_umap_embeddings() # Function for plotting UMAP embeddings
│
├── thtt.py                    # Training and testing
│   ├── Data preparation       # Prepare data for training
│   ├── Model training         # Train models on different datasets
│   └── Cross-testing         # Test models across different datasets
│
└── thtt_utils.py             # Training utilities
    ├── convert_channel_angle_delay()  # Convert channel data format
    └── train_val_test_split()        # Split data for training
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

## Dependencies

- numpy
- matplotlib
- cuml (for UMAP)
- deepmimo (for ray tracing data)
- sionna (for stochastic channel models)

## Usage

1. Configure data generation parameters in `data_gen.py`
2. Generate/load data using `load_data_matrices()`
3. Visualize data topology using `topology.py`:
   - Compute UMAP embeddings
   - Visualize with/without outliers
   - Perform partial UMAP fitting
4. Train and test models using `thtt.py`:
   - Prepare data using utilities from `thtt_utils.py`
   - Train models on different datasets
   - Cross-test models across datasets