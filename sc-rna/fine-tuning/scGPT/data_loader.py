"""
Data loading and preprocessing utilities for scGPT fine-tuning with NPZ files.
"""

import gc
import os
import pickle
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from scgpt import logger


class NPZDataLoader:
    """Handles loading and preprocessing of NPZ single-cell data for fine-tuning."""
    
    def __init__(self, data_config):
        self.data_config = data_config
        
    def load_label_data(self) -> Dict:
        """Load label data from pickle file."""
        with open(self.data_config.label_pkl_path, 'rb') as f:
            label_data = pickle.load(f)
        return label_data
    
    def build_subset_log_dataframe(self, label_data: Dict, n_class0: int, n_class1: int, 
                                 seed: int = 42) -> pd.DataFrame:
        """
        Select a balanced subset of training donors and return donor-level DataFrame.

        Args:
            label_data (dict): Dictionary containing 'train' key with 'SubID' and target column lists.
            n_class0 (int): Number of donors to sample with label 0 (-1 for all).
            n_class1 (int): Number of donors to sample with label 1 (-1 for all).
            seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: Donor-level DataFrame with columns: SubID, target_column, used (always True).
        """
        # Cell-level to donor-level
        donor_df = pd.DataFrame({
            "SubID": label_data['train']['SubID'],
            self.data_config.phen_column: label_data['train'][self.data_config.phen_column]
        }).drop_duplicates(subset='SubID')

        # Sample donors per class
        class0_donors = donor_df[donor_df[self.data_config.phen_column] == 0]
        class1_donors = donor_df[donor_df[self.data_config.phen_column] == 1]
        
        if n_class0 == -1:
            selected_class0 = class0_donors
        else:
            selected_class0 = class0_donors.sample(n=n_class0, random_state=seed)
            
        if n_class1 == -1:
            selected_class1 = class1_donors
        else:
            selected_class1 = class1_donors.sample(n=n_class1, random_state=seed)

        selected_donors = pd.concat([selected_class0, selected_class1])
        selected_donors['used'] = True

        return selected_donors.reset_index(drop=True)
    
    def get_donor_files(self) -> Tuple[List[str], Dict[str, str]]:
        """Get list of donor files and create donor ID to path mapping."""
        donor_files = [f for f in os.listdir(self.data_config.data_path) 
                      if f.endswith("_aligned.npz")]
        
        donor_id_to_path = {}
        for f in donor_files:
            donor_id = os.path.basename(f).replace("_aligned.npz", "")
            donor_id_to_path[donor_id] = os.path.join(self.data_config.data_path, f)
            
        return donor_files, donor_id_to_path
    
    def create_train_val_split(self, selected_donors: pd.DataFrame, 
                             donor_id_to_path: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Create train/validation split at donor level."""
        # Filter to donors that have corresponding files
        available_donors = [d for d in selected_donors['SubID'] if d in donor_id_to_path]
        
        if not available_donors:
            raise ValueError("No donors found with corresponding NPZ files")
        
        # Donor-level train/val split
        train_donor_ids, val_donor_ids = train_test_split(
            available_donors,
            test_size=self.data_config.val_split_ratio,
            random_state=self.data_config.random_seed,
            shuffle=True
        )
        
        train_paths = [donor_id_to_path[d] for d in train_donor_ids]
        val_paths = [donor_id_to_path[d] for d in val_donor_ids]
        
        logger.info(f"Donor-level split — Train donors: {len(train_donor_ids)} | Val donors: {len(val_donor_ids)}")
        
        return train_paths, val_paths
    
    def create_label_map(self, label_data: Dict) -> Dict[str, int]:
        """Create mapping from donor ID to label."""
        return dict(zip(
            label_data['train'][self.data_config.sample_column], 
            label_data['train'][self.data_config.phen_column]
        ))
    
    def prepare_data_splits(self) -> Tuple[List[str], List[str], Dict[str, int], pd.DataFrame]:
        """
        Prepare train/validation data splits with NPZ files.
        
        Returns:
            Tuple of (train_paths, val_paths, label_map, subset_log_df)
        """
        # Load label data
        label_data = self.load_label_data()
        
        # Create subset selection if specified
        if self.data_config.n_class0 == -1 and self.data_config.n_class1 == -1:
            # Use all donors
            train_df = pd.DataFrame({
                "SubID": label_data['train']['SubID'],
                self.data_config.phen_column: label_data['train'][self.data_config.phen_column]
            }).drop_duplicates(subset='SubID')
            train_df['used'] = True
            subset_log_df = train_df.copy()
            logger.info(f"Using all {len(train_df)} donors for training.")
        else:
            # Use subset
            subset_log_df = self.build_subset_log_dataframe(
                label_data, 
                self.data_config.n_class0, 
                self.data_config.n_class1, 
                seed=self.data_config.random_seed
            )
            logger.info(f"Using subset with {len(subset_log_df)} donors.")
        
        # Save subset log
        subset_log_df.to_csv(self.data_config.subset_log_csv, index=False)
        
        # Get donor files
        donor_files, donor_id_to_path = self.get_donor_files()
        
        # Create train/val split
        train_paths, val_paths = self.create_train_val_split(subset_log_df, donor_id_to_path)
        
        # Create label map
        label_map = self.create_label_map(label_data)
        
        return train_paths, val_paths, label_map, subset_log_df


def normalize_expression(row: np.ndarray, target_sum: float = 1e4) -> Tuple[np.ndarray, float, float]:
    """
    Normalize gene expression row and compute additional features.
    
    Args:
        row: Gene expression counts
        target_sum: Target sum for normalization
        
    Returns:
        Tuple of (normalized_expression, resolution_token, log_total)
    """
    total_count = row.sum()
    norm_expr = np.log1p(row / total_count * target_sum)  # CPM + log1p normalization
    
    log_total = np.log10(total_count + 1e-8)  # Safe log
    res_token = log_total + 5.0  # Resolution token (matching tgthighres = a5)
    
    return norm_expr, res_token, log_total
