"""
Configuration template for scGPT fine-tuning.

Copy the paths below and update them in train.py at the top of the file.
"""

# =====================================================
# EXAMPLE FILE PATHS CONFIGURATION
# =====================================================

# Example configuration for your data:

# Data paths
LABEL_PKL_PATH = "/home/user/data/my_labels.pkl"              # Path to pickle file with labels
NPZ_DATA_PATH = "/home/user/data/donors/"                     # Directory containing donor NPZ files  
SUBSET_LOG_CSV = "./my_subset_log.csv"                        # Log file for subset selection

# Model paths  
PRETRAINED_MODEL_PATH = "/home/user/models/scGPT_model.pt"    # Pre-trained scGPT model
VOCAB_FILE_PATH = "/home/user/models/vocab.json"              # Vocabulary file

# Output paths
SAVE_DIR = "./my_results"                                      # Directory to save results and models
WANDB_PROJECT = "my-scgpt-project"                            # W&B project name

# Data configuration
PHENOTYPE_COLUMN = "cell_type"                                 # Column name for labels in your data
SAMPLE_ID_COLUMN = "donor_id"                                 # Column name for donor IDs
N_CLASSES = 3                                                  # Number of classification classes

# Subset selection (set to -1 to use all donors)
N_CLASS0_DONORS = 50                                           # Number of class 0 donors (-1 for all)
N_CLASS1_DONORS = 50                                           # Number of class 1 donors (-1 for all)

# Training settings
VALIDATION_SPLIT = 0.25                                        # Fraction of donors for validation
RANDOM_SEED = 123                                              # Random seed for reproducibility
BATCH_SIZE = 16                                                # Training batch size
LEARNING_RATE = 5e-5                                           # Learning rate
MAX_EPOCHS = 100                                               # Maximum training epochs

# Model settings
MAX_SEQUENCE_LENGTH = 2000                                     # Maximum sequence length
MASK_RATIO = 0.20                                              # Masking ratio for training
INPUT_STYLE = "log1p"                                          # Input normalization style
INPUT_EMB_STYLE = "category"                                   # Input embedding style

# =====================================================
# QUICK SETUP CHECKLIST
# =====================================================

"""
Before running training:

1. Update LABEL_PKL_PATH to your labels file
2. Update NPZ_DATA_PATH to your NPZ files directory  
3. Update PRETRAINED_MODEL_PATH to your model file
4. Update VOCAB_FILE_PATH to your vocabulary file
5. Set PHENOTYPE_COLUMN to match your label column name
6. Set SAMPLE_ID_COLUMN to match your donor ID column
7. Set N_CLASSES to your number of classes
8. Adjust training parameters as needed
9. Run: python train.py

Your NPZ files should be named like: donor1_aligned.npz, donor2_aligned.npz, etc.

Your labels.pkl should contain:
{
    'train': {
        'SubID': ['donor1', 'donor2', ...],
        'your_phenotype_column': [0, 1, 0, ...]
    }
}
"""
