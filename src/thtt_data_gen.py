#%% Import Modules
import os
import pickle

from data_gen import DataConfig, load_data_matrices

# Data paths
DATA_FOLDER = 'data'
MATRIX_NAME = 'data_matrices_50k_complex_all2.pkl'
MAT_PATH = os.path.join(DATA_FOLDER, MATRIX_NAME)

# Channel Models
ch_models = ['CDL-C', 'UMa'] # 'UMa!param!asu_campus_3p5'
rt_scens = ['asu_campus_3p5', 'city_0_newyork_3p5']
models = rt_scens + ch_models

NT = 32
NC = 16

def pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
#%% [SIONNA ENV] Load and Prepare Data

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
data_cfg.rt_sample_trimming = False  # to enforce same number of samples for all models

# Load data
data_matrices = load_data_matrices(models, data_cfg)

#%% [SIONNA ENV] Save matrices
os.makedirs(DATA_FOLDER, exist_ok=True)
pickle_save(data_matrices, MAT_PATH)
