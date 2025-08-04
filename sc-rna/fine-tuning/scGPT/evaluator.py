"""
Evaluation utilities for scGPT fine-tuning.
"""

import torch
import wandb
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scgpt import logger

try:
    from torchmetrics import Accuracy, Precision, Recall, F1Score
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    logger.warning("torchmetrics not available, using sklearn metrics")


class Evaluator:
    """Handles model evaluation for scGPT fine-tuning."""
    
    def __init__(self, model_config, device, vocab, num_classes: int):
        self.model_config = model_config
        self.device = device
        self.vocab = vocab
        self.num_classes = num_classes
        
    def evaluate(self, model: nn.Module, loader: DataLoader, criteria: dict,
                epoch: int, return_raw: bool = False) -> Union[Tuple[float, float], np.ndarray]:
        """
        Evaluate the model on the evaluation data.
        
        Args:
            model: The model to evaluate
            loader: Data loader for evaluation
            criteria: Dictionary of loss criteria
            epoch: Current epoch number
            return_raw: Whether to return raw predictions
            
        Returns:
            Either (loss, error_rate) tuple or raw predictions array
        """
        model.eval()
        
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        all_celltype_labels = []

        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)

                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.model_config.pad_token])
                
                with torch.cuda.amp.autocast(enabled=hasattr(self, 'amp') and self.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if (self.model_config.INPUT_BATCH_LABELS or 
                                                    hasattr(model, 'domain_spec_batchnorm')) else None,
                        CLS=True,  # Always do classification for evaluation
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                    )
                    
                    output_values = output_dict["cls_output"]
                    loss = criteria['cls'](output_values, celltype_labels)

                    if self.model_config.DAB and "dab_output" in output_dict:
                        loss_dab = criteria['dab'](output_dict["dab_output"], batch_labels)
                        total_dab += loss_dab.item() * len(input_gene_ids)

                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                
                preds = output_values.argmax(1).cpu().numpy()
                labels = celltype_labels.cpu().numpy()
                
                predictions.append(preds)
                all_celltype_labels.append(labels)
        
        # Concatenate predictions and labels
        predictions = np.concatenate(predictions, axis=0)
        all_celltype_labels = np.concatenate(all_celltype_labels, axis=0)
        
        # Calculate metrics
        accuracy = accuracy_score(all_celltype_labels, predictions)
        precision = precision_score(all_celltype_labels, predictions, average="macro", zero_division=0)
        recall = recall_score(all_celltype_labels, predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(all_celltype_labels, predictions, average="macro", zero_division=0)

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}"
        )

        # Log to wandb
        wandb.log({
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + total_dab) / total_num,
            "valid/accuracy": accuracy,
            "valid/precision": precision,
            "valid/recall": recall,
            "valid/f1": macro_f1,
            "epoch": epoch,
        })

        if return_raw:
            return predictions

        return total_loss / total_num, total_error / total_num

    def evaluate_torchmetrics(self, model: nn.Module, loader: DataLoader, criteria: dict,
                             epoch: int, return_raw: bool = False) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Evaluate using torchmetrics for GPU-accelerated metrics.
        
        Args:
            model: The model to evaluate
            loader: Data loader for evaluation
            criteria: Dictionary of loss criteria
            epoch: Current epoch number
            return_raw: Whether to return raw predictions and labels
            
        Returns:
            Either loss value or (predictions, labels) tuple
        """
        if not TORCHMETRICS_AVAILABLE:
            logger.warning("torchmetrics not available, falling back to sklearn")
            return self.evaluate(model, loader, criteria, epoch, return_raw)
        
        model.eval()

        # Initialize metrics on device
        acc_metric = Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)
        prec_metric = Precision(task='multiclass', average="macro", num_classes=self.num_classes).to(self.device)
        recall_metric = Recall(task='multiclass', average="macro", num_classes=self.num_classes).to(self.device)
        f1_metric = F1Score(task='multiclass', average="macro", num_classes=self.num_classes).to(self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_dab = torch.tensor(0.0, device=self.device)
        total_num = 0

        # For return_raw
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in loader:
                gene_ids = batch_data["gene_ids"].to(self.device)
                values = batch_data["values"].to(self.device)
                batch_lbl = batch_data["batch_labels"].to(self.device)
                cell_lbl = batch_data["celltype_labels"].to(self.device)

                mask = gene_ids.eq(self.vocab[self.model_config.pad_token])

                with torch.cuda.amp.autocast(enabled=hasattr(self, 'amp') and self.amp):
                    out = model(
                        gene_ids, values,
                        src_key_padding_mask=mask,
                        batch_labels=(batch_lbl if (self.model_config.INPUT_BATCH_LABELS or 
                                                  hasattr(model, 'domain_spec_batchnorm')) else None),
                        CLS=True, CCE=False, MVC=False, ECS=False,
                        do_sample=False,
                    )

                    logits = out["cls_output"]
                    loss = criteria['cls'](logits, cell_lbl)
                    total_loss += loss * gene_ids.size(0)

                    if self.model_config.DAB and "dab_output" in out:
                        dab_loss = criteria['dab'](out["dab_output"], batch_lbl)
                        total_dab += dab_loss * gene_ids.size(0)

                # Update metrics on GPU
                preds = logits.argmax(dim=1)
                acc_metric.update(preds, cell_lbl)
                prec_metric.update(preds, cell_lbl)
                recall_metric.update(preds, cell_lbl)
                f1_metric.update(preds, cell_lbl)

                total_num += gene_ids.size(0)

                if return_raw:
                    all_preds.append(preds.cpu())
                    all_labels.append(cell_lbl.cpu())

        # Compute metrics
        accuracy = acc_metric.compute().item()
        precision = prec_metric.compute().item()
        recall = recall_metric.compute().item()
        macro_f1 = f1_metric.compute().item()

        logger.info(
            f"Acc: {accuracy:.3f}, Prec: {precision:.3f}, "
            f"Rec: {recall:.3f}, F1: {macro_f1:.3f}"
        )

        # Log metrics
        wandb.log({
            "valid/loss": (total_loss / total_num).item(),
            "valid/dab": (total_dab / total_num).item(),
            "valid/accuracy": accuracy,
            "valid/precision": precision,
            "valid/recall": recall,
            "valid/f1": macro_f1,
            "epoch": epoch,
        })

        if return_raw:
            preds_arr = torch.cat(all_preds).numpy()
            labels_arr = torch.cat(all_labels).numpy()
            return preds_arr, labels_arr

        return (total_loss / total_num).item()

    def test(self, model: nn.Module, adata, test_loader: Optional[DataLoader] = None,
            criteria: dict = None, prepare_test_data_func = None, 
            prepare_dataloader_func = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on test data.
        
        Args:
            model: The model to test
            adata: AnnData object (if test_loader is None)
            test_loader: Optional pre-prepared test loader
            criteria: Loss criteria
            prepare_test_data_func: Function to prepare test data
            prepare_dataloader_func: Function to prepare dataloader
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        if test_loader is None:
            if prepare_test_data_func is None or prepare_dataloader_func is None:
                raise ValueError("Need either test_loader or prepare functions")
                
            # Prepare test data using provided function
            test_data_pt = prepare_test_data_func(adata)
            test_loader = prepare_dataloader_func(
                test_data_pt, 
                batch_size=64,  # Default eval batch size
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=False,
            )

        model.eval()
        predictions = self.evaluate(
            model,
            loader=test_loader,
            criteria=criteria,
            epoch=0,  # Not used for testing
            return_raw=True,
        )

        # Get true labels from adata if available
        if hasattr(adata, 'obs'):
            phen = getattr(self.model_config, 'phen_column', 'celltype')
            celltypes_labels = adata.obs[f"{phen}_id"].values
        else:
            celltypes_labels = None

        return predictions, celltypes_labels


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }
    
    return metrics
