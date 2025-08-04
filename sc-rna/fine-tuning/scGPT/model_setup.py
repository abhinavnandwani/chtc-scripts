"""
Model initialization and setup utilities for scGPT fine-tuning.
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np
from torch import nn

from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import masked_mse_loss
from scgpt import logger


class ModelSetup:
    """Manages model initialization, loading, and setup for NPZ-based fine-tuning."""
    
    def __init__(self, model_config, device):
        self.model_config = model_config
        self.device = device
        self.vocab = None
        self.model = None
        
    def setup_vocabulary(self, vocab: GeneVocab):
        """Setup vocabulary for the model."""
        self.vocab = vocab
        # Add special tokens if not present
        for token in self.model_config.special_tokens:
            if token not in vocab:
                vocab.append_token(token)
        
        vocab.set_default_index(vocab[self.model_config.pad_token])
        logger.info(f"Vocabulary setup completed with {len(vocab)} tokens")
        
    def load_pretrained_model(self, model_file: str) -> TransformerModel:
        """Load pre-trained scGPT model."""
        logger.info(f"Loading pre-trained model from: {model_file}")
        
        model = torch.load(model_file, map_location=self.device)
        
        # Check if model is wrapped in state dict
        if hasattr(model, 'state_dict'):
            # Create new model instance and load state dict
            # This may need to be adjusted based on your specific model architecture
            logger.warning("Model loaded as state dict - may need architecture setup")
        
        model.to(self.device)
        self.model = model
        
        logger.info("Pre-trained model loaded successfully")
        return model
    
    def setup_for_finetuning(self, model: TransformerModel, n_cls: int, n_batch: int) -> TransformerModel:
        """Setup model for fine-tuning with classification head."""
        
        # Enable CLS mode for classification
        model.cls_mode = True
        
        # Setup classification head if needed
        if hasattr(model, 'classifier') and model.classifier is not None:
            # Adjust classifier for the number of classes
            if model.classifier.out_features != n_cls:
                logger.info(f"Adjusting classifier from {model.classifier.out_features} to {n_cls} classes")
                model.classifier = nn.Linear(model.classifier.in_features, n_cls)
        else:
            # Create new classifier
            logger.info(f"Creating new classifier for {n_cls} classes")
            model.classifier = nn.Linear(model.d_model, n_cls)
        
        # Setup batch classifier if needed for domain adaptation
        if hasattr(model, 'batch_classifier') and n_batch > 1:
            if model.batch_classifier is None or model.batch_classifier.out_features != n_batch:
                logger.info(f"Setting up batch classifier for {n_batch} batches")
                model.batch_classifier = nn.Linear(model.d_model, n_batch)
        
        # Move updated components to device
        model.to(self.device)
        
        # Set to training mode
        model.train()
        
        logger.info("Model setup for fine-tuning completed")
        return model
    
    def get_gene_ids_for_vocab(self) -> np.ndarray:
        """Get gene IDs for tokenization based on vocabulary."""
        if self.vocab is None:
            raise ValueError("Vocabulary must be setup before getting gene IDs")
            
        # Create gene list from vocabulary (excluding special tokens)
        special_tokens = set(self.model_config.special_tokens)
        genes = [token for token in self.vocab.get_itos() if token not in special_tokens]
        
        # Get IDs for genes
        gene_ids = np.array(self.vocab(genes), dtype=int)
        
        logger.info(f"Generated gene IDs for {len(gene_ids)} genes")
        return gene_ids
    
    def freeze_pretrained_weights(self, model: TransformerModel, freeze_backbone: bool = False):
        """Optionally freeze pre-trained weights."""
        if freeze_backbone:
            logger.info("Freezing backbone weights")
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'batch_classifier' not in name:
                    param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")


class ModelManager:
    """Legacy model manager for compatibility."""
    
    def __init__(self, fine_tuning_config, model_config, save_dir: Path):
        self.ft_config = fine_tuning_config
        self.model_config = model_config
        self.save_dir = save_dir
        
    def setup_model_parameters(self, vocab_len: int, num_types: int, 
                             num_batch_types: int) -> dict:
        """Setup model parameters based on configuration."""
        # Set up value encoding parameters
        if self.model_config.input_emb_style == "category":
            mask_value = self.ft_config.n_bins + 1
            pad_value = self.ft_config.n_bins
            n_input_bins = self.ft_config.n_bins + 2
        else:
            mask_value = -1
            pad_value = -2
            n_input_bins = self.ft_config.n_bins
            
        params = {
            'ntokens': vocab_len,
            'embsize': self.ft_config.layer_size,
            'nhead': self.ft_config.nhead,
            'd_hid': self.ft_config.layer_size,
            'nlayers': self.ft_config.nlayers,
            'nlayers_cls': len(self.ft_config.cls_hlayers) + 1,
            'n_cls': num_types if self.model_config.CLS else 1,
            'ndims': self.ft_config.cls_hlayers,
            'dropout': self.ft_config.dropout,
            'pad_token': self.model_config.pad_token,
            'pad_value': pad_value,
            'mask_value': mask_value,
            'n_input_bins': n_input_bins,
            'num_batch_types': num_batch_types,
        }
        
        return params
    
    def load_pretrained_config(self, model_dir: Path) -> Tuple[dict, Path, Path, Path]:
        """Load configuration from pretrained model."""
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        
        # Copy vocab file to save directory
        shutil.copy(vocab_file, self.save_dir / "vocab.json")
        
        # Load model configuration
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
            
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        
        return model_configs, model_file, vocab_file, model_config_file
    
    def create_model(self, vocab, model_params: dict, model_configs: Optional[dict] = None) -> TransformerModel:
        """Create TransformerModel with given parameters."""
        # Override parameters with pretrained model config if available
        if model_configs is not None:
            model_params['embsize'] = model_configs["embsize"]
            model_params['nhead'] = model_configs["nheads"] 
            model_params['d_hid'] = model_configs["d_hid"]
            model_params['nlayers'] = model_configs["nlayers"]
            model_params['nlayers_cls'] = model_configs["n_layers_cls"]
        
        model = TransformerModel(
            model_params['ntokens'],
            model_params['embsize'],
            model_params['nhead'],
            model_params['d_hid'],
            nlayers=model_params['nlayers'],
            nlayers_cls=model_params['nlayers_cls'],
            n_cls=model_params['n_cls'],
            ndims=model_params['ndims'],
            vocab=vocab,
            dropout=model_params['dropout'],
            pad_token=model_params['pad_token'],
            pad_value=model_params['pad_value'],
            do_mvc=self.model_config.MLM,  # Set based on config
            do_dab=self.model_config.DAB,
            use_batch_labels=self.model_config.INPUT_BATCH_LABELS,
            num_batch_labels=model_params['num_batch_types'],
            domain_spec_batchnorm=self.ft_config.DSBN,
            input_emb_style=self.model_config.input_emb_style,
            n_input_bins=model_params['n_input_bins'],
            cell_emb_style=self.model_config.cell_emb_style,
            mvc_decoder_style=self.model_config.mvc_decoder_style,
            ecs_threshold=self.ft_config.ecs_thres,
            explicit_zero_prob=self.model_config.explicit_zero_prob,
            use_fast_transformer=self.ft_config.fast_transformer,
            fast_transformer_backend=self.model_config.fast_transformer_backend,
            pre_norm=self.ft_config.pre_norm,
        )
        
        return model
    
    def load_pretrained_weights(self, model: TransformerModel, model_file: Path) -> TransformerModel:
        """Load pretrained weights into model."""
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except Exception as e:
            logger.warning(f"Could not load all params, loading compatible ones: {e}")
            # Only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model
    
    def apply_parameter_freezing(self, model: TransformerModel) -> Tuple[int, int]:
        """Apply parameter freezing strategy."""
        # Count parameters before freezing
        pre_freeze_count = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
        )
        
        # Freeze all parameters first
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer layers
        unfreeze_last_n = self.ft_config.unfreeze_last_n
        for layer in model.transformer_encoder.layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Count parameters after freezing
        post_freeze_count = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()
        )
        
        logger.info(f"Total Pre freeze Params: {pre_freeze_count}")
        logger.info(f"Total Post freeze Params: {post_freeze_count}")
        
        return pre_freeze_count, post_freeze_count
    
    def setup_discriminator(self, embsize: int, num_batch_types: int) -> Optional[AdversarialDiscriminator]:
        """Setup adversarial discriminator if needed."""
        if self.model_config.ADV:
            discriminator = AdversarialDiscriminator(
                d_model=embsize,
                n_cls=num_batch_types,
            )
            return discriminator
        return None
    
    def setup_criteria(self) -> dict:
        """Setup loss criteria."""
        criteria = {
            'mse': masked_mse_loss,
            'cls': nn.CrossEntropyLoss(),
            'dab': nn.CrossEntropyLoss(),
        }
        
        if self.model_config.ADV:
            criteria['adv'] = nn.CrossEntropyLoss()
            
        return criteria
    
    def setup_optimizers(self, model: TransformerModel, 
                        discriminator: Optional[AdversarialDiscriminator] = None) -> dict:
        """Setup optimizers and schedulers."""
        optimizers = {}
        schedulers = {}
        
        # Main optimizer
        optimizers['main'] = torch.optim.AdamW(
            model.parameters(), 
            lr=self.ft_config.lr, 
            eps=1e-4 if self.ft_config.amp else 1e-8
        )
        schedulers['main'] = torch.optim.lr_scheduler.StepLR(
            optimizers['main'], 
            self.model_config.schedule_interval, 
            gamma=self.ft_config.schedule_ratio
        )
        
        # DAB optimizer (if separate optimization is needed)
        if self.model_config.DAB and hasattr(self.model_config, 'DAB_separate_optim') and self.model_config.DAB_separate_optim:
            optimizers['dab'] = torch.optim.Adam(model.parameters(), lr=self.ft_config.lr)
            schedulers['dab'] = torch.optim.lr_scheduler.StepLR(
                optimizers['dab'], 
                self.model_config.schedule_interval, 
                gamma=self.ft_config.schedule_ratio
            )
        
        # Adversarial optimizers
        if self.model_config.ADV and discriminator is not None:
            optimizers['encoder'] = torch.optim.Adam(
                model.parameters(), lr=self.model_config.lr_ADV
            )
            schedulers['encoder'] = torch.optim.lr_scheduler.StepLR(
                optimizers['encoder'], 
                self.model_config.schedule_interval, 
                gamma=self.ft_config.schedule_ratio
            )
            
            optimizers['discriminator'] = torch.optim.Adam(
                discriminator.parameters(), lr=self.model_config.lr_ADV
            )
            schedulers['discriminator'] = torch.optim.lr_scheduler.StepLR(
                optimizers['discriminator'], 
                self.model_config.schedule_interval, 
                gamma=self.ft_config.schedule_ratio
            )
        
        return {'optimizers': optimizers, 'schedulers': schedulers}


def setup_value_encoding(n_bins: int, input_emb_style: str) -> Tuple[int, int, int]:
    """Setup value encoding parameters based on input style."""
    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins
    
    return mask_value, pad_value, n_input_bins
