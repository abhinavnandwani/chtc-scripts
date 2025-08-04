"""
Dataset and DataLoader utilities for scGPT fine-tuning with NPZ files.
"""

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
from typing import Dict, List, Tuple

from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import logger
from data_loader import normalize_expression


class GeneExpressionDataset(Dataset):
    """PyTorch Dataset for gene expression data from NPZ files."""
    
    def __init__(self, 
                 file_paths: List[str], 
                 label_map: Dict[str, int], 
                 gene_ids: np.ndarray,
                 vocab,
                 max_seq_len: int = 1200,
                 pad_token: str = "<pad>",
                 mask_ratio: float = 0.0,
                 mask_value: int = -1,
                 pad_value: int = -2,
                 include_zero_gene: bool = True,
                 append_cls: bool = True,
                 input_style: str = "normed_raw",
                 input_emb_style: str = "continuous",
                 amp: bool = False):
        """
        Initialize the dataset.
        
        Args:
            file_paths: List of paths to NPZ files
            label_map: Mapping from donor ID to label
            gene_ids: Array of gene IDs for vocabulary
            vocab: Vocabulary object
            max_seq_len: Maximum sequence length
            pad_token: Padding token
            mask_ratio: Ratio of genes to mask during training
            mask_value: Value to use for masked genes
            pad_value: Value to use for padding
            include_zero_gene: Whether to include zero expression genes
            append_cls: Whether to append CLS token
            input_style: Input normalization style
            input_emb_style: Input embedding style
            amp: Whether to use automatic mixed precision
        """
        self.file_paths = file_paths
        self.label_map = label_map
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.include_zero_gene = include_zero_gene
        self.append_cls = append_cls
        self.input_style = input_style
        self.input_emb_style = input_emb_style
        self.amp = amp
        
        # Cache for data
        self._cache = {}
        
        # Build index mapping
        self._build_index_mapping()
        
    def _build_index_mapping(self):
        """Build mapping from dataset index to (file_idx, cell_idx)."""
        self.index_mapping = []
        self.file_cell_counts = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            # Load sparse matrix to get shape
            sparse_matrix = load_npz(file_path)
            n_cells = sparse_matrix.shape[0]
            self.file_cell_counts.append(n_cells)
            
            # Add mappings for all cells in this file
            for cell_idx in range(n_cells):
                self.index_mapping.append((file_idx, cell_idx))
                
        logger.info(f"Dataset built with {len(self.index_mapping)} cells from {len(self.file_paths)} files")
    
    def _get_donor_id_from_path(self, file_path: str) -> str:
        """Extract donor ID from file path."""
        return os.path.basename(file_path).replace("_aligned.npz", "")
    
    def _load_file_data(self, file_idx: int) -> Tuple[np.ndarray, int]:
        """Load data from file and cache it."""
        if file_idx in self._cache:
            return self._cache[file_idx]
            
        file_path = self.file_paths[file_idx]
        
        # Load sparse matrix and convert to dense
        sparse_matrix = load_npz(file_path)
        expression_data = sparse_matrix.toarray().astype(np.float32)
        
        # Get label for this donor
        donor_id = self._get_donor_id_from_path(file_path)
        label = self.label_map.get(donor_id, 0)  # Default to 0 if not found
        
        # Cache the data
        self._cache[file_idx] = (expression_data, label)
        
        return expression_data, label
    
    def __len__(self) -> int:
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        file_idx, cell_idx = self.index_mapping[idx]
        
        # Load file data
        expression_data, label = self._load_file_data(file_idx)
        
        # Get expression for this cell
        cell_expression = expression_data[cell_idx]
        
        # Normalize expression
        if self.input_style == "normed_raw":
            norm_expr, res_token, log_total = normalize_expression(cell_expression)
            input_values = norm_expr
        else:
            # For other styles, use raw counts
            input_values = cell_expression
            res_token = 0.0
            log_total = np.log10(cell_expression.sum() + 1e-8)
        
        # Tokenize and pad
        tokenized_data = tokenize_and_pad_batch(
            input_values[None, :],  # Add batch dimension
            self.gene_ids,
            max_len=self.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene,
        )
        
        # Extract from batch format
        gene_ids = tokenized_data["gene_ids"][0]
        values = tokenized_data["values"][0]
        
        # Apply masking if needed
        if self.mask_ratio > 0:
            gene_ids, values = random_mask_value(
                gene_ids, values, mask_ratio=self.mask_ratio, mask_value=self.mask_value, pad_value=self.pad_value
            )
        
        # Prepare output
        sample = {
            "gene_ids": torch.from_numpy(gene_ids.copy()).long(),
            "values": torch.from_numpy(values.copy()).float(),
            "target_values": torch.from_numpy(values.copy()).float(),  # For reconstruction loss
            "celltype_labels": torch.tensor(label, dtype=torch.long),
            "batch_labels": torch.tensor(0, dtype=torch.long),  # Default batch
        }
        
        # Add resolution token if using continuous embedding
        if self.input_emb_style == "continuous":
            sample["resolution_token"] = torch.tensor(res_token, dtype=torch.float32)
        
        return sample


class SeqDataset(Dataset):
    """Legacy dataset class for compatibility."""
    
    def __init__(self, data, vocab, mask_ratio=0.0, mask_value=-1, pad_value=-2, 
                 include_zero_gene=True, append_cls=True, max_seq_len=1200, pad_token="<pad>"):
        self.data = data
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.include_zero_gene = include_zero_gene
        self.append_cls = append_cls
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        
    def __len__(self):
        return len(self.data["celltype_labels"])
    
    def __getitem__(self, idx):
        return {
            "gene_ids": torch.from_numpy(self.data["gene_ids"][idx].copy()).long(),
            "values": torch.from_numpy(self.data["values"][idx].copy()).float(),
            "target_values": torch.from_numpy(self.data["target_values"][idx].copy()).float(),
            "celltype_labels": torch.tensor(self.data["celltype_labels"][idx], dtype=torch.long),
            "batch_labels": torch.tensor(self.data["batch_labels"][idx], dtype=torch.long),
        }


def prepare_dataloader(dataset: Dataset, 
                      batch_size: int,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      pin_memory: bool = True,
                      drop_last: bool = False) -> DataLoader:
    """Prepare DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_datasets_from_npz(train_paths: List[str],
                            val_paths: List[str],
                            label_map: Dict[str, int],
                            gene_ids: np.ndarray,
                            vocab,
                            config) -> Tuple[GeneExpressionDataset, GeneExpressionDataset]:
    """
    Create train and validation datasets from NPZ files.
    
    Args:
        train_paths: List of training file paths
        val_paths: List of validation file paths
        label_map: Mapping from donor ID to label
        gene_ids: Array of gene IDs
        vocab: Vocabulary object
        config: Configuration object with model settings
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Training dataset with masking
    train_dataset = GeneExpressionDataset(
        file_paths=train_paths,
        label_map=label_map,
        gene_ids=gene_ids,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        pad_token=config.pad_token,
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
        include_zero_gene=config.include_zero_gene,
        append_cls=config.append_cls,
        input_style=config.input_style,
        input_emb_style=config.input_emb_style,
        amp=config.amp
    )
    
    # Validation dataset without masking
    val_dataset = GeneExpressionDataset(
        file_paths=val_paths,
        label_map=label_map,
        gene_ids=gene_ids,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        pad_token=config.pad_token,
        mask_ratio=0.0,  # No masking for validation
        mask_value=config.mask_value,
        pad_value=config.pad_value,
        include_zero_gene=config.include_zero_gene,
        append_cls=config.append_cls,
        input_style=config.input_style,
        input_emb_style=config.input_emb_style,
        amp=config.amp
    )
    
    return train_dataset, val_dataset


def create_dataloaders_from_npz(train_paths: List[str],
                               val_paths: List[str],
                               label_map: Dict[str, int],
                               gene_ids: np.ndarray,
                               vocab,
                               config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from NPZ files.
    
    Args:
        train_paths: List of training file paths
        val_paths: List of validation file paths
        label_map: Mapping from donor ID to label
        gene_ids: Array of gene IDs
        vocab: Vocabulary object
        config: Configuration object
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset, val_dataset = create_datasets_from_npz(
        train_paths, val_paths, label_map, gene_ids, vocab, config
    )
    
    # Create dataloaders
    train_dataloader = prepare_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 0,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = prepare_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 0,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader

import os
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from scgpt import SubsetsBatchSampler
from scgpt.tokenizer import random_mask_value


class SeqDataset(Dataset):
    """Dataset class for tokenized sequence data."""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_data(tokenized_train: Dict, tokenized_valid: Dict, 
                train_batch_labels: np.ndarray, valid_batch_labels: np.ndarray,
                train_celltype_labels: np.ndarray, valid_celltype_labels: np.ndarray,
                mask_ratio: float, mask_value: int, pad_value: int,
                sort_seq_batch: bool = False, epoch: int = 1) -> tuple:
    """
    Prepare training and validation data with masking.
    
    Args:
        tokenized_train: Tokenized training data
        tokenized_valid: Tokenized validation data
        train_batch_labels: Training batch labels
        valid_batch_labels: Validation batch labels
        train_celltype_labels: Training cell type labels
        valid_celltype_labels: Validation cell type labels
        mask_ratio: Ratio of values to mask
        mask_value: Value to use for masking
        pad_value: Value to use for padding
        sort_seq_batch: Whether to sort by batch
        epoch: Current epoch number
        
    Returns:
        Tuple of train and validation data dictionaries
    """
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    print(
        f"Random masking at epoch {epoch:3d}, ratio of masked values in train: "
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}"
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: Optional[int] = None,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    """
    Prepare PyTorch DataLoader from processed data.
    
    Args:
        data_pt: Dictionary of tensors
        batch_size: Batch size
        shuffle: Whether to shuffle data
        intra_domain_shuffle: Whether to shuffle within domains
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of workers for data loading
        per_seq_batch_sample: Whether to use per-sequence batch sampling
        
    Returns:
        PyTorch DataLoader
    """
    if num_workers is None:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # Find indices of samples in each sequence batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def prepare_test_data(adata, gene_ids, vocab, input_layer_key: str, 
                     phen: str, mask_ratio: float, mask_value: int, 
                     pad_value: int, max_seq_len: int, pad_token: str,
                     include_zero_gene: bool = True) -> Dict[str, torch.Tensor]:
    """
    Prepare test data for inference.
    
    Args:
        adata: AnnData object
        gene_ids: Gene ID array
        vocab: Vocabulary object
        input_layer_key: Key for input layer
        phen: Phenotype column name
        mask_ratio: Masking ratio
        mask_value: Mask value
        pad_value: Padding value
        max_seq_len: Maximum sequence length
        pad_token: Padding token
        include_zero_gene: Whether to include zero genes
        
    Returns:
        Dictionary of test data tensors
    """
    from scipy.sparse import issparse
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs[f"{phen}_id"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    
    return test_data_pt
