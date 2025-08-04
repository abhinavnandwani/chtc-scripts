"""
Hyperparameter configuration for scGPT fine-tuning.
"""

import os
from pathlib import Path
from typing import Dict, Any


class FineTuningConfig:
    """Configuration class for fine-tuning hyperparameters and settings."""
    
    def __init__(self):
        # Basic settings
        self.seed = 0
        self.dataset_name = "mssm"
        self.do_train = True
        
        # Model loading
        self.load_model = "../pretrained_models/scGPT_full_body"
        
        # Training hyperparameters
        self.mask_ratio = 0.0
        self.epochs = 5
        self.n_bins = 51
        self.lr = 1e-4
        self.batch_size = 64
        self.eval_batch_size = 64
        
        # Model architecture
        self.layer_size = 128
        self.nlayers = 12  # number of nn.TransformerEncoderLayer
        self.nhead = 4  # number of heads in nn.MultiheadAttention
        self.dropout = 0.2
        
        # Training objectives
        self.MVC = False  # Masked value prediction for cell embedding
        self.ecs_thres = 0.0  # Elastic cell similarity objective
        self.dab_weight = 0.0
        
        # Training settings
        self.schedule_ratio = 0.9
        self.save_eval_interval = 5
        self.fast_transformer = True
        self.pre_norm = False
        self.amp = True  # Automatic Mixed Precision
        self.include_zero_gene = False
        self.freeze = False
        self.DSBN = False  # Domain-spec batchnorm
        
        # Additional settings
        self.cls_hlayers = [256, 64]
        self.unfreeze_last_n = 3
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def get_wandb_defaults(self) -> Dict[str, Any]:
        """Get wandb-compatible hyperparameter defaults."""
        return {
            "seed": self.seed,
            "dataset_name": self.dataset_name,
            "do_train": self.do_train,
            "load_model": self.load_model,
            "mask_ratio": self.mask_ratio,
            "epochs": self.epochs,
            "n_bins": self.n_bins,
            "MVC": self.MVC,
            "ecs_thres": self.ecs_thres,
            "dab_weight": self.dab_weight,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "layer_size": self.layer_size,
            "nlayers": self.nlayers,
            "nhead": self.nhead,
            "dropout": self.dropout,
            "schedule_ratio": self.schedule_ratio,
            "save_eval_interval": self.save_eval_interval,
            "fast_transformer": self.fast_transformer,
            "pre_norm": self.pre_norm,
            "amp": self.amp,
            "include_zero_gene": self.include_zero_gene,
            "freeze": self.freeze,
            "DSBN": self.DSBN,
        }


class DataConfig:
    """Configuration for data paths and processing."""
    
    def __init__(self):
        # Data paths - Update these for your setup
        self.sample_column = 'SubID'
        self.phen_column = 'c02x'
        self.label_pkl_path = "../../c02x_split_seed42.pkl"  # Path to label pickle file
        self.data_path = "/path/to/donor/npz/files/"  # Path to donor *_aligned.npz files
        
        # Data processing settings
        self.val_split_ratio = 0.2  # Validation split ratio
        self.random_seed = 42
        
        # Subset selection (use -1 for all donors)
        self.n_class0 = -1  # Number of donors with class 0
        self.n_class1 = -1  # Number of donors with class 1
        
        # Save directory
        self.save_dir_base = "/home/ubuntu/scGPT/scGPT_finetuning/save/"
        self.subset_log_csv = "subset_log.csv"


class ModelConfig:
    """Configuration for model settings."""
    
    def __init__(self):
        # Input/output settings
        self.pad_token = "<pad>"
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]
        self.mask_value = "auto"
        self.max_seq_len = 3001
        
        # Style settings
        self.input_style = "binned"  # "normed_raw", "log1p", or "binned"
        self.output_style = "binned"
        self.input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
        self.cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
        self.mvc_decoder_style = "inner product"
        
        # Training objectives
        self.MLM = False  # Masked language modeling
        self.CLS = True   # Cell type classification
        self.ADV = False  # Adversarial training
        self.CCE = False  # Contrastive cell embedding
        self.ECS = False  # Elastic cell similarity
        self.DAB = False  # Domain adaptation
        self.INPUT_BATCH_LABELS = False
        
        # Training fractions
        self.MLM_FRACTION = 1
        self.MVC_FRACTION = 1
        self.CLS_FRACTION = 1
        
        # Advanced settings
        self.explicit_zero_prob = False
        self.do_sample_in_train = False
        self.per_seq_batch_sample = False
        self.adv_E_delay_epochs = 0
        self.adv_D_delay_epochs = 0
        self.lr_ADV = 1e-3
        self.schedule_interval = 1
        self.log_interval = 100
        self.do_eval_scib_metrics = True
        self.fast_transformer_backend = "flash"
