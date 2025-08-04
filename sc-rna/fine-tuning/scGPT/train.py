"""
Main training script for scGPT fine-tuning with NPZ data files.
"""

import os
import warnings
import torch
import wandb
import numpy as np
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

from scgpt import logger
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# Import modular components
from hyperparameters import FineTuningConfig
from data_loader import NPZDataLoader
from dataset import create_dataloaders_from_npz
from model_setup import ModelSetup
from trainer import FineTuningTrainer
from evaluator import ModelEvaluator
from helpers import (
    set_seed, 
    setup_directories, 
    save_hyperparameters,
    configure_device
)

# =====================================================
# FILE PATHS CONFIGURATION - MODIFY THESE PATHS
# =====================================================

# Data paths
LABEL_PKL_PATH = "/path/to/your/labels.pkl"                    # Path to pickle file with labels
NPZ_DATA_PATH = "/path/to/your/npz/files/"                    # Directory containing donor NPZ files
SUBSET_LOG_CSV = "./subset_selection_log.csv"                 # Log file for subset selection

# Model paths  
PRETRAINED_MODEL_PATH = "/path/to/pretrained/model.pt"        # Pre-trained scGPT model
VOCAB_FILE_PATH = "/path/to/vocab.json"                       # Vocabulary file

# Output paths
SAVE_DIR = "./results"                                         # Directory to save results and models
WANDB_PROJECT = "scgpt-finetuning"                            # W&B project name

# Data configuration
PHENOTYPE_COLUMN = "phenotype"                                 # Column name for labels in your data
SAMPLE_ID_COLUMN = "SubID"                                     # Column name for donor IDs
N_CLASSES = 2                                                  # Number of classification classes

# Subset selection (set to -1 to use all donors)
N_CLASS0_DONORS = -1                                           # Number of class 0 donors (-1 for all)
N_CLASS1_DONORS = -1                                           # Number of class 1 donors (-1 for all)

# Training settings
VALIDATION_SPLIT = 0.2                                         # Fraction of donors for validation
RANDOM_SEED = 42                                               # Random seed for reproducibility
BATCH_SIZE = 32                                                # Training batch size
LEARNING_RATE = 1e-4                                           # Learning rate
MAX_EPOCHS = 50                                                # Maximum training epochs

# Model settings
MAX_SEQUENCE_LENGTH = 1200                                     # Maximum sequence length
MASK_RATIO = 0.15                                              # Masking ratio for training
INPUT_STYLE = "normed_raw"                                     # Input normalization style
INPUT_EMB_STYLE = "continuous"                                 # Input embedding style

# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================


def main():
    """
    Main training function for scGPT fine-tuning.
    
    To use this script:
    1. Modify the paths in the "FILE PATHS CONFIGURATION" section above
    2. Adjust any training hyperparameters as needed
    3. Run: python train.py
    """
    
    # ==================== CONFIGURATION ====================
    config = FineTuningConfig()
    
    # Override configuration with consolidated paths and settings
    config.data_config.label_pkl_path = LABEL_PKL_PATH
    config.data_config.data_path = NPZ_DATA_PATH
    config.data_config.subset_log_csv = SUBSET_LOG_CSV
    config.data_config.phen_column = PHENOTYPE_COLUMN
    config.data_config.sample_column = SAMPLE_ID_COLUMN
    config.data_config.n_cls = N_CLASSES
    config.data_config.n_class0 = N_CLASS0_DONORS
    config.data_config.n_class1 = N_CLASS1_DONORS
    config.data_config.val_split_ratio = VALIDATION_SPLIT
    config.data_config.random_seed = RANDOM_SEED
    
    config.model_config.model_file = PRETRAINED_MODEL_PATH
    config.model_config.vocab_file = VOCAB_FILE_PATH
    config.model_config.max_seq_len = MAX_SEQUENCE_LENGTH
    config.model_config.mask_ratio = MASK_RATIO
    config.model_config.batch_size = BATCH_SIZE
    config.model_config.input_style = INPUT_STYLE
    config.model_config.input_emb_style = INPUT_EMB_STYLE
    
    config.train_config.save_dir = SAVE_DIR
    config.train_config.wandb_project = WANDB_PROJECT
    config.train_config.random_seed = RANDOM_SEED
    config.train_config.learning_rate = LEARNING_RATE
    config.train_config.max_epochs = MAX_EPOCHS
    config.train_config.batch_size = BATCH_SIZE
    
    # Print configuration summary
    print("=" * 60)
    print("scGPT Fine-tuning Configuration")
    print("=" * 60)
    print(f"Data Path: {NPZ_DATA_PATH}")
    print(f"Labels Path: {LABEL_PKL_PATH}")
    print(f"Model Path: {PRETRAINED_MODEL_PATH}")
    print(f"Vocab Path: {VOCAB_FILE_PATH}")
    print(f"Save Dir: {SAVE_DIR}")
    print(f"Phenotype Column: {PHENOTYPE_COLUMN}")
    print(f"Classes: {N_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print("=" * 60)
    
    # Set random seeds
    set_seed(config.train_config.random_seed)
    
    # Configure device
    device = configure_device(config.train_config.device)
    
    # Setup directories
    setup_directories(config.train_config.save_dir)
    
    # Save hyperparameters
    save_hyperparameters(config, config.train_config.save_dir)
    
    logger.info(f"Starting fine-tuning with config: {config.train_config.save_dir}")
    
    # ==================== WANDB SETUP ====================
    if config.train_config.do_train and config.train_config.log_wandb:
        wandb.init(
            project=config.train_config.wandb_project,
            config=config.to_dict(),
            name=f"{config.data_config.phen_column}_{config.train_config.run_name}",
            tags=[config.data_config.phen_column, "npz_data", "fine_tuning"]
        )
        logger.info(f"W&B logging enabled. Project: {config.train_config.wandb_project}")
    
    # ==================== DATA PREPARATION ====================
    logger.info("Loading and preparing data...")
    
    # Initialize data loader
    data_loader = NPZDataLoader(config.data_config)
    
    # Prepare data splits
    train_paths, val_paths, label_map, subset_log_df = data_loader.prepare_data_splits()
    
    logger.info(f"Data splits prepared:")
    logger.info(f"  Train files: {len(train_paths)}")
    logger.info(f"  Validation files: {len(val_paths)}")
    logger.info(f"  Total donors: {len(subset_log_df)}")
    
    # Log class distribution
    class_counts = subset_log_df[config.data_config.phen_column].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")
    
    # ==================== MODEL SETUP ====================
    logger.info("Setting up model and vocabulary...")
    
    model_setup = ModelSetup(config.model_config, device)
    
    # Load vocabulary
    vocab = GeneVocab.from_file(config.model_config.vocab_file)
    model_setup.setup_vocabulary(vocab)
    
    # Load pre-trained model
    model = model_setup.load_pretrained_model(config.model_config.model_file)
    
    # Setup for fine-tuning
    model = model_setup.setup_for_finetuning(
        model, 
        n_cls=config.data_config.n_cls,
        n_batch=1  # Single batch since we're using NPZ files
    )
    
    # Get gene IDs for tokenization
    gene_ids = model_setup.get_gene_ids_for_vocab()
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Gene IDs for tokenization: {len(gene_ids)}")
    
    # ==================== DATASET CREATION ====================
    logger.info("Creating datasets and dataloaders...")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders_from_npz(
        train_paths=train_paths,
        val_paths=val_paths,
        label_map=label_map,
        gene_ids=gene_ids,
        vocab=vocab,
        config=config.model_config
    )
    
    logger.info(f"Dataloaders created:")
    logger.info(f"  Train batches: {len(train_dataloader)}")
    logger.info(f"  Validation batches: {len(val_dataloader)}")
    
    # ==================== TRAINING SETUP ====================
    if config.train_config.do_train:
        logger.info("Setting up trainer...")
        
        trainer = FineTuningTrainer(
            model=model,
            config=config.train_config,
            device=device,
            vocab=vocab
        )
        
        # ==================== TRAINING LOOP ====================
        logger.info("Starting training...")
        
        best_val_loss, best_model = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            save_dir=config.train_config.save_dir
        )
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        # Save best model
        best_model_path = Path(config.train_config.save_dir) / "best_model.pt"
        torch.save(best_model, best_model_path)
        logger.info(f"Best model saved to: {best_model_path}")
        
        # ==================== EVALUATION ====================
        if config.train_config.do_eval:
            logger.info("Starting evaluation...")
            
            evaluator = ModelEvaluator(
                model=best_model,
                device=device,
                config=config.eval_config
            )
            
            # Evaluate on validation set
            val_metrics = evaluator.evaluate(val_dataloader, split_name="validation")
            
            logger.info("Validation Results:")
            for metric, value in val_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Log to wandb
            if config.train_config.log_wandb:
                wandb.log({"final_" + k: v for k, v in val_metrics.items()})
    
    # ==================== INFERENCE ONLY ====================
    elif config.train_config.do_eval:
        logger.info("Loading model for evaluation...")
        
        # Load best model if available
        model_path = Path(config.train_config.save_dir) / "best_model.pt"
        if model_path.exists():
            model = torch.load(model_path, map_location=device)
            logger.info(f"Loaded model from: {model_path}")
        else:
            logger.warning("No trained model found. Using pre-trained model for evaluation.")
        
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            config=config.eval_config
        )
        
        # Evaluate on validation set
        val_metrics = evaluator.evaluate(val_dataloader, split_name="validation")
        
        logger.info("Evaluation Results:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # ==================== CLEANUP ====================
    if config.train_config.log_wandb:
        wandb.finish()
    
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
