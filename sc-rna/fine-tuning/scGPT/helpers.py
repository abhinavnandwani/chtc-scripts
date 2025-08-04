"""
Utility functions for scGPT fine-tuning.
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import wandb

from scgpt.utils import set_seed
from scgpt import logger


def setup_logging_and_directories(config) -> Path:
    """
    Setup logging and create save directories.
    
    Args:
        config: Configuration object with dataset_name
        
    Returns:
        Path to save directory
    """
    dataset_name = config.dataset_name
    save_dir = Path(f"/home/ubuntu/scGPT/scGPT_finetuning/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Save to {save_dir}")
    
    # Setup file logging
    from scgpt.utils import add_file_handler
    add_file_handler(logger, save_dir / "run.log")
    
    return save_dir


def setup_wandb(config) -> wandb.sdk.wandb_run.Run:
    """
    Initialize wandb for experiment tracking.
    
    Args:
        config: Configuration object
        
    Returns:
        wandb run object
    """
    run = wandb.init(
        config=config.get_wandb_defaults() if hasattr(config, 'get_wandb_defaults') else config.to_dict(),
        project="scGPT-finetuning",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    
    wandb_config = wandb.config
    print("Wandb config:", wandb_config)
    
    return run


def setup_device_and_environment(config) -> torch.device:
    """
    Setup device and environment variables.
    
    Args:
        config: Configuration object with seed
        
    Returns:
        Device object
    """
    # Set random seed
    set_seed(config.seed)
    
    # Setup environment
    os.environ["KMP_WARNINGS"] = "off"
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    return device


def save_model_checkpoint(model: torch.nn.Module, save_dir: Path, 
                         epoch: Optional[int] = None, is_best: bool = False) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_dir: Directory to save to
        epoch: Current epoch (for filename)
        is_best: Whether this is the best model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_best:
        filename = save_dir / "best_model.pt"
    elif epoch is not None:
        filename = save_dir / f"model_epoch_{epoch}_{timestamp}.pt"
    else:
        filename = save_dir / f"model_{timestamp}.pt"
    
    torch.save(model.state_dict(), filename)
    logger.info(f"Saved model to {filename}")


def log_parameter_counts(model: torch.nn.Module, stage: str = "") -> Dict[str, int]:
    """
    Log parameter counts for the model.
    
    Args:
        model: Model to analyze
        stage: Stage identifier (e.g., "pre_freeze", "post_freeze")
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"{stage} Parameter counts:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Frozen: {frozen_params:,}")
    
    return {
        f"{stage}_total_params": total_params,
        f"{stage}_trainable_params": trainable_params,
        f"{stage}_frozen_params": frozen_params,
    }


def update_learning_rate(optimizers: Dict[str, torch.optim.Optimizer], 
                        schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler]) -> None:
    """
    Update learning rates using schedulers.
    
    Args:
        optimizers: Dictionary of optimizers
        schedulers: Dictionary of schedulers
    """
    for name, scheduler in schedulers.items():
        if name in optimizers:
            scheduler.step()


def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get a summary of the model architecture.
    
    Args:
        model: Model to summarize
        
    Returns:
        Dictionary with model information
    """
    summary = {
        'model_type': type(model).__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'num_layers': getattr(model, 'nlayers', 'unknown'),
        'embedding_size': getattr(model, 'd_model', 'unknown'),
        'num_heads': getattr(model, 'nhead', 'unknown'),
    }
    
    return summary


def create_run_config(ft_config, model_config, data_config) -> Dict[str, Any]:
    """
    Create a comprehensive configuration dictionary for the run.
    
    Args:
        ft_config: Fine-tuning configuration
        model_config: Model configuration  
        data_config: Data configuration
        
    Returns:
        Combined configuration dictionary
    """
    run_config = {
        'fine_tuning': ft_config.to_dict() if hasattr(ft_config, 'to_dict') else vars(ft_config),
        'model': vars(model_config),
        'data': vars(data_config),
        'timestamp': datetime.now().isoformat(),
    }
    
    return run_config


def validate_paths(data_config) -> bool:
    """
    Validate that all required data paths exist.
    
    Args:
        data_config: Data configuration object
        
    Returns:
        True if all paths exist, False otherwise
    """
    required_paths = [
        'meta_path',
        'data_path', 
        'save_file_path',
        'gene_var_path',
        'clinical_meta_path',
        'meta_obs_path'
    ]
    
    missing_paths = []
    
    for path_attr in required_paths:
        if hasattr(data_config, path_attr):
            path = getattr(data_config, path_attr)
            if not os.path.exists(path):
                missing_paths.append(f"{path_attr}: {path}")
        else:
            missing_paths.append(f"Missing attribute: {path_attr}")
    
    if missing_paths:
        logger.error("Missing or invalid paths:")
        for path in missing_paths:
            logger.error(f"  {path}")
        return False
    
    return True


def estimate_memory_usage(batch_size: int, seq_len: int, embed_dim: int, 
                         num_layers: int, precision: str = "fp32") -> Dict[str, float]:
    """
    Estimate GPU memory usage for the model.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        precision: Precision type ("fp32", "fp16", "amp")
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = 4 if precision == "fp32" else 2
    
    # Rough estimates
    model_params = embed_dim * embed_dim * num_layers * 4  # Simplified
    activation_memory = batch_size * seq_len * embed_dim * num_layers
    
    model_memory_gb = (model_params * bytes_per_param) / (1024**3)
    activation_memory_gb = (activation_memory * bytes_per_param) / (1024**3)
    total_memory_gb = model_memory_gb + activation_memory_gb
    
    return {
        'model_memory_gb': model_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_estimated_gb': total_memory_gb,
    }


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
            
        return False
