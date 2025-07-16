"""Configuration management for channel model training.

This module provides configuration classes to manage model parameters,
training settings, and data paths in a centralized way.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import os
import copy

@dataclass
class ModelConfig:
    """Configuration for model architecture and training.
    
    This class centralizes all parameters needed for model creation and training,
    including support for fine-tuning models.
    
    Example usage:
        # Create base config
        base_config = ModelConfig(encoded_dim=32)
        
        # Create fine-tuning config
        finetune_config = base_config.for_finetuning(
            source_model="UMa",
            num_epochs=15
        )
    """
    # Model architecture
    encoded_dim: int = 32
    n_refine_nets: int = 5
    n_taps: int = 100
    n_antennas: int = 64
    
    # Training parameters
    train_batch_size: int = 32
    test_batch_size: int = 1024
    num_epochs: int = 200
    learning_rate: float = 1e-2
    n_train_samples: Optional[int] = None
    n_val_samples: Optional[int] = None
    n_test_samples: Optional[int] = None
    seed: int = 42
    
    # Directory structure
    dataset_main_folder: str = 'channel_datasets'
    
    # Fine-tuning parameters
    is_finetuning: bool = False
    source_model: Optional[str] = None
    
    def __post_init__(self):
        """Validate fine-tuning configuration."""
        if self.is_finetuning and self.source_model is None:
            raise ValueError("source_model must be provided when is_finetuning=True")
            
        # Ensure folders exist
        os.makedirs(self.dataset_main_folder, exist_ok=True)
        if self.is_finetuning:
            os.makedirs(self.dataset_main_folder + '_finetuned', exist_ok=True)

    def get_dataset_folder(self, model: str) -> str:
        """Get path to dataset folder for a model."""
        return os.path.join(self.dataset_main_folder, f'model_{model}')

    def get_model_path(self, model: str) -> str:
        """Get path to save/load model weights.
        
        When fine-tuning:
        - For loading: Uses source model path from base folder
        - For saving: Uses fine-tuned folder with source model name
        - Format: model_{target}_{source}.pth
        
        Otherwise:
        - Uses base folder for both loading and saving
        - Format: model_{model}.pth
        """
        if self.is_finetuning:
            # When fine-tuning, we load from source model and save to fine-tuned folder
            if model == self.source_model:
                # When loading source model, use base folder
                folder = self.dataset_main_folder
                return os.path.join(folder, f"model_{model}.pth")
            else:
                # When saving fine-tuned model, use fine-tuned folder
                folder = self.dataset_main_folder + '_finetuned'
                return os.path.join(folder, f"model_{self.source_model}_{model}.pth")
        else:
            # Normal training uses base folder
            folder = self.dataset_main_folder
            return os.path.join(folder, f"model_{model}.pth")

    def clone(self, **kwargs) -> 'ModelConfig':
        """Create a copy of this config with some parameters changed."""
        return ModelConfig(**{**asdict(self), **kwargs})

    def for_finetuning(self,
                      source_model: str,
                      num_epochs: int = 15,
                      learning_rate: Optional[float] = None,
                      n_train_samples: Optional[int] = None) -> 'ModelConfig':
        """Create a configuration for fine-tuning.
        
        Args:
            source_model: Name of the model to load pretrained weights from
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning (None = use base config's rate)
            n_train_samples: Number of samples to use (None = use base config)
            
        Returns:
            New ModelConfig instance configured for fine-tuning
        """
        return self.clone(
            is_finetuning=True,
            source_model=source_model,
            num_epochs=num_epochs,
            learning_rate=learning_rate if learning_rate is not None else self.learning_rate,
            n_train_samples=n_train_samples if n_train_samples is not None else self.n_train_samples
        )
        