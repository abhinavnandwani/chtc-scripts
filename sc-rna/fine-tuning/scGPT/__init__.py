"""
Modularized scGPT Fine-tuning Package

This package provides a modular framework for fine-tuning scGPT models
on cell-type annotation and other downstream tasks.

Main components:
- hyperparameters: Configuration classes for hyperparameters, data, and model settings
- data_loader: Data loading and preprocessing utilities
- dataset: PyTorch dataset and dataloader utilities
- model_setup: Model setup, initialization, and management
- trainer: Training loop and optimization utilities  
- evaluator: Model evaluation and metrics computation
- helpers: Helper functions and utilities

Usage:
    python train_npz.py
    # Run the main training script for NPZ data
    
Or use individual components:
    from hyperparameters import FineTuningConfig
    from data_loader import NPZDataLoader
    from model_setup import ModelSetup
    # etc.
"""

__version__ = "1.0.0"

# Import main components for easy access
from hyperparameters import FineTuningConfig, DataConfig, ModelConfig
from data_loader import NPZDataLoader
from dataset import GeneExpressionDataset, create_dataloaders_from_npz
from model_setup import ModelSetup
from trainer import FineTuningTrainer
from evaluator import ModelEvaluator
from helpers import set_seed, setup_directories, configure_device

__all__ = [
    'FineTuningConfig', 'DataConfig', 'ModelConfig',
    'NPZDataLoader', 'GeneExpressionDataset', 'create_dataloaders_from_npz',
    'ModelSetup', 'FineTuningTrainer', 'ModelEvaluator',
    'set_seed', 'setup_directories', 'configure_device'
]
