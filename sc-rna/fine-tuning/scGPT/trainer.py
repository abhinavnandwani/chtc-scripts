"""
Training utilities for scGPT fine-tuning.
"""

import time
import warnings
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional

from scgpt import logger


class Trainer:
    """Handles the training loop for scGPT fine-tuning."""
    
    def __init__(self, model_config, device, vocab, scaler):
        self.model_config = model_config
        self.device = device
        self.vocab = vocab
        self.scaler = scaler
        
    def train_epoch(self, model: nn.Module, loader: DataLoader, epoch: int,
                   criteria: dict, optimizers: dict, 
                   discriminator: Optional[nn.Module] = None) -> None:
        """
        Train the model for one epoch.
        
        Args:
            model: The transformer model
            loader: Training data loader
            epoch: Current epoch number
            criteria: Dictionary of loss criteria
            optimizers: Dictionary of optimizers
            discriminator: Optional discriminator for adversarial training
        """
        model.train()
        
        # Initialize loss trackers
        total_loss = 0.0
        total_mse = 0.0
        total_cls = 0.0
        total_cce = 0.0
        total_mvc = 0.0
        total_ecs = 0.0
        total_dab = 0.0
        total_adv_E = 0.0
        total_adv_D = 0.0
        total_zero_log_prob = 0.0
        total_mvc_zero_log_prob = 0.0
        total_error = 0.0
        
        start_time = time.time()
        num_batches = len(loader)
        
        for batch, batch_data in enumerate(loader):
            try:
                # Move data to device
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)

                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.model_config.pad_token])
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if (self.model_config.INPUT_BATCH_LABELS or 
                                                    hasattr(model, 'domain_spec_batchnorm')) else None,
                        CLS=self.model_config.CLS,
                        CCE=self.model_config.CCE,
                        MVC=self.model_config.MLM,  # Using MLM for MVC
                        ECS=self.model_config.ECS,
                        do_sample=self.model_config.do_sample_in_train,
                    )

                    # Calculate losses
                    loss = 0.0
                    metrics_to_log = {}
                    
                    masked_positions = input_values.eq(-1)  # Default mask value
                    
                    # MLM loss
                    if self.model_config.MLM and "mlm_output" in output_dict:
                        loss_mse = criteria['mse'](
                            output_dict["mlm_output"], target_values, masked_positions
                        )
                        loss += loss_mse * self.model_config.MLM_FRACTION
                        metrics_to_log["train/mse"] = loss_mse.item()
                        total_mse += loss_mse.item()
                    
                    # Classification loss
                    if self.model_config.CLS and "cls_output" in output_dict:
                        loss_cls = criteria['cls'](output_dict["cls_output"], celltype_labels)
                        loss += loss_cls * self.model_config.CLS_FRACTION
                        metrics_to_log["train/cls"] = loss_cls.item()
                        total_cls += loss_cls.item()

                        # Calculate error rate
                        error_rate = 1 - (
                            (output_dict["cls_output"].argmax(1) == celltype_labels)
                            .sum().item()
                        ) / celltype_labels.size(0)
                        total_error += error_rate
                    
                    # CCE loss
                    if self.model_config.CCE and "loss_cce" in output_dict:
                        loss_cce = 10 * output_dict["loss_cce"]
                        loss += loss_cce
                        metrics_to_log["train/cce"] = loss_cce.item()
                        total_cce += loss_cce.item()
                    
                    # MVC loss
                    if self.model_config.MLM and "mvc_output" in output_dict:
                        loss_mvc = criteria['mse'](
                            output_dict["mvc_output"], target_values, masked_positions
                        )
                        loss += loss_mvc * self.model_config.MVC_FRACTION
                        metrics_to_log["train/mvc"] = loss_mvc.item()
                        total_mvc += loss_mvc.item()
                    
                    # ECS loss
                    if self.model_config.ECS and "loss_ecs" in output_dict:
                        loss_ecs = 10 * output_dict["loss_ecs"]
                        loss += loss_ecs
                        metrics_to_log["train/ecs"] = loss_ecs.item()
                        total_ecs += loss_ecs.item()
                    
                    # DAB loss
                    if self.model_config.DAB and "dab_output" in output_dict:
                        loss_dab = criteria['dab'](output_dict["dab_output"], batch_labels)
                        loss += loss_dab * getattr(self.model_config, 'dab_weight', 1.0)
                        metrics_to_log["train/dab"] = loss_dab.item()
                        total_dab += loss_dab.item()

                # Backward pass
                model.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizers['main'])

                # Gradient clipping with warning handling
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0,
                        error_if_nonfinite=False if self.scaler.is_enabled() else True,
                    )
                    if len(w) > 0:
                        logger.warning(
                            f"Found infinite gradient. Scale: {self.scaler.get_scale()}. "
                            "This warning can be ignored if it stops after autoscaling."
                        )

                self.scaler.step(optimizers['main'])
                self.scaler.update()

                # Adversarial training
                if self.model_config.ADV and discriminator is not None:
                    self._adversarial_training_step(
                        model, discriminator, input_gene_ids, input_values,
                        src_key_padding_mask, batch_labels, criteria,
                        optimizers, epoch, metrics_to_log
                    )

                # Log metrics
                wandb.log(metrics_to_log)
                total_loss += loss.item()

                # Periodic logging
                if batch % self.model_config.log_interval == 0 and batch > 0:
                    self._log_training_progress(
                        epoch, batch, num_batches, total_loss, total_mse, total_cls,
                        total_error, start_time, optimizers['main']
                    )
                    # Reset counters
                    total_loss = total_mse = total_cls = total_error = 0
                    start_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in batch {batch}: {e}")
                continue

    def _adversarial_training_step(self, model, discriminator, input_gene_ids, input_values,
                                 src_key_padding_mask, batch_labels, criteria, optimizers,
                                 epoch, metrics_to_log):
        """Perform adversarial training step."""
        # Rerun model for adversarial training
        output_dict = model(
            input_gene_ids, input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if (self.model_config.INPUT_BATCH_LABELS or 
                                        hasattr(model, 'domain_spec_batchnorm')) else None,
            CLS=self.model_config.CLS,
            CCE=self.model_config.CCE,
            MVC=self.model_config.MLM,
            ECS=self.model_config.ECS,
            do_sample=self.model_config.do_sample_in_train,
        )

        # Train discriminator
        loss_adv_D = criteria['adv'](
            discriminator(output_dict["cell_emb"].detach()), batch_labels
        )
        if epoch > self.model_config.adv_D_delay_epochs:
            discriminator.zero_grad()
            loss_adv_D.backward()
            optimizers['discriminator'].step()

        # Train encoder (adversarially)
        loss_adv_E = -criteria['adv'](
            discriminator(output_dict["cell_emb"]), batch_labels
        )
        if epoch > self.model_config.adv_E_delay_epochs:
            model.zero_grad()
            discriminator.zero_grad()
            loss_adv_E.backward()
            optimizers['encoder'].step()

        metrics_to_log.update({
            "train/adv_D": loss_adv_D.item(),
            "train/adv_E": loss_adv_E.item(),
        })

    def _log_training_progress(self, epoch, batch, num_batches, total_loss, total_mse,
                             total_cls, total_error, start_time, optimizer):
        """Log training progress."""
        lr = optimizer.param_groups[0]['lr']
        ms_per_batch = (time.time() - start_time) * 1000 / self.model_config.log_interval
        cur_loss = total_loss / self.model_config.log_interval
        cur_mse = total_mse / self.model_config.log_interval
        cur_cls = total_cls / self.model_config.log_interval if self.model_config.CLS else 0.0
        cur_error = total_error / self.model_config.log_interval

        log_msg = (
            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
            f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
            f"loss {cur_loss:5.2f} | "
        )
        
        if self.model_config.MLM:
            log_msg += f"mse {cur_mse:5.2f} | "
        if self.model_config.CLS:
            log_msg += f"cls {cur_cls:5.2f} | err {cur_error:5.2f} | "

        logger.info(log_msg)


def define_wandb_metrics():
    """Define wandb metrics for tracking."""
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")
