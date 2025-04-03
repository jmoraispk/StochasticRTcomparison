# Stochastic RT Comparison

This project compares stochastic channel models with ray tracing models using UMAP visualization and machine learning techniques.

## Project Structure

```
├── data_gen.py                 # Core data generation and preparation
│   ├── DataConfig              # Configuration class
│   ├── load_data_matrices()    # Load data for different models
│   ├── prepare_data_for_analysis()  # Prepare data for analysis
│   └── outlier detection functions  # Detect and remove outliers
│
├── sionna_ch_gen.py            # Sionna-specific channel generation
│   └── Channel generation functions
│
├── topology.py                 # UMAP visualization and analysis
│   ├── plot_umap_embeddings()  # Visualize UMAP embeddings
│   └── main()                  # Example usage
│
└── thtt.py                     # Training and testing
    ├── train_models()          # Train models for each area
    ├── cross_test_models()     # Cross-test models across datasets
    ├── plot_training_results() # Plot training progress
    └── plot_test_matrix()      # Visualize test results
```

## Data Flow

```
data_gen.py ──────┐
                  ├─> topology.py (UMAP visualization)
sionna_ch_gen.py ─┘
                  └─> thtt.py (Training and testing)
```

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- umap-learn
- sionna
- deepmimo
- dataset_utils

## Usage

1. Generate data using `data_gen.py` and `sionna_ch_gen.py`
2. Visualize data topology using `topology.py`
3. Train and test models using `thtt.py`