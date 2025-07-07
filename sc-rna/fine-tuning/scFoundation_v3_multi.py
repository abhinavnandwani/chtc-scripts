import os
import sys
import signal
import gc
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import load_npz
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

###### for distributed training #####
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
####################################

import argparse
from tqdm import tqdm
from load import load_model_frommmf

# ===== CONFIG =====
LABEL_PKL_PATH     = "../../c02x_split_seed42.pkl"
PRETRAINED_CKPT    = "./models/models.ckpt"
SAVE_DIR           = "./models/finetuned_c02x"
UNFREEZE_LAST_N    = 2
EPOCHS             = 5
BATCH_SIZE         = 10  # per GPU
LEARNING_RATE      = 1e-3
RANDOM_SEED        = 42

# Global model reference for cleanup
model = None

# ===== CLEANUP FUNCTION =====
def cleanup():
    global model
    print("\n[INFO] Cleaning up resources...")
    if model is not None:
        del model
        model = None
        print("[INFO] Model reference deleted.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] CUDA memory cache cleared.")
    gc.collect()
    print("[INFO] CPU memory released (garbage collected).")
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        print(f"[INFO] Post-cleanup GPU memory: allocated={mem_alloc:.2f}MB, reserved={mem_reserved:.2f}MB")

import atexit
atexit.register(cleanup)

# ===== DATASET CLASS =====
class GeneExpressionDataset(Dataset):
    def __init__(self, file_paths, label_map):
        self.samples = []
        self.memory = {}
        for path in file_paths:
            donor_id = os.path.basename(path).replace("_aligned.npz", "")
            label = label_map.get(donor_id)
            if label is None:
                continue
            matrix = load_npz(path)
            self.memory[donor_id] = matrix
            for i in range(matrix.shape[0]):
                self.samples.append((donor_id, i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        donor_id, row_idx, label = self.samples[idx]
        row = self.memory[donor_id].getrow(row_idx).toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ===== MODEL CLASS =====
class FineTuneClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes=1, unfreeze_last_n=2):
        super().__init__()
        self.model, self.model_config = load_model_frommmf(ckpt_path)
        self.token_emb = self.model.token_emb
        self.pos_emb = self.model.pos_emb
        self.encoder = self.model.encoder

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze last N transformer layers
        for layer in self.encoder.transformer_encoder[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.norm = nn.BatchNorm1d(self.model_config['encoder']['hidden_dim'], affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_config['encoder']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, batch):
        x = batch['x']
        value_mask = x > 0
        gene_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.token_emb(x.unsqueeze(-1).float(), output_weight=0)
        x += self.pos_emb(gene_ids)
        x = self.encoder(x, padding_mask=~value_mask)
        x = torch.max(x, dim=1).values
        x = self.norm(x)
        return self.classifier(x)

# ===== TRAINING FUNCTION =====
def train_model(model, train_loader, test_loader, optimizer, criterion, device, epoch, rank, wandb_enabled):
    model.train()
    total_loss, preds, truths = 0, [], []

    # Use tqdm for rank 0 only
    train_pbar = tqdm(train_loader, desc="Training", disable=(rank != 0))

    for batch in train_pbar:
        x, y = batch
        x, y = x.to(device), y.to(device).unsqueeze(1)
        logits = model({'x': x})
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds.extend((logits > 0).long().cpu().tolist())
        truths.extend(y.cpu().long().tolist())

        if rank == 0:
            avg_loss = total_loss / len(preds)
            train_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    if rank == 0:
        train_acc = balanced_accuracy_score(truths, preds)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Balanced Acc: {train_acc:.4f}")

    # ===== EVALUATION =====
    model.eval()
    val_preds, val_truths = [], []

    val_pbar = tqdm(test_loader, desc="Validation", disable=(rank != 0))

    with torch.no_grad():
        for batch in val_pbar:
            x, y = batch
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model({'x': x})
            val_preds.extend((logits > 0).long().cpu().tolist())
            val_truths.extend(y.cpu().long().tolist())

    if rank == 0:
        val_acc = balanced_accuracy_score(val_truths, val_preds)
        print(f"[Epoch {epoch+1}] Val Balanced Acc: {val_acc:.4f}")
        if wandb_enabled:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "train/balanced_acc": train_acc,
                "val/balanced_acc": val_acc
            })

# ===== MAIN DDP FUNCTION =====
def run_ddp(rank, world_size, args):
    # Setup distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize wandb only on rank 0
    wandb_enabled = False
    if rank == 0:
        try:
            import wandb
            wandb.login(key="f04957e341167ac5452921a251b0921fedd3558b")
            wandb.init(
                project="scFoundation_finetune",
                config={
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "unfreeze_last_n": UNFREEZE_LAST_N
                },
                mode="online"
            )
            wandb_enabled = True
        except Exception as e:
            print(f"[WARN] W&B disabled on rank 0: {e}")

    # Load label mapping
    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)

    train_ids = set(label_data['train']['SubID'])
    test_ids = set(label_data['test']['SubID'])
    label_map = {
        **dict(zip(label_data['train']['SubID'], label_data['train']['c02x'])),
        **dict(zip(label_data['test']['SubID'], label_data['test']['c02x']))
    }

    donor_files = [f for f in os.listdir(args.data_path) if f.endswith("_aligned.npz")]
    donor_ids = [os.path.splitext(f.replace("_aligned", ""))[0] for f in donor_files]

    train_paths = [os.path.join(args.data_path, f) for f, did in zip(donor_files, donor_ids) if did in train_ids]
    test_paths = [os.path.join(args.data_path, f) for f, did in zip(donor_files, donor_ids) if did in test_ids]


    train_paths = train_paths[:10]
    test_paths = test_paths[:2]


    # Dataset & DataLoader with DistributedSampler
    train_ds = GeneExpressionDataset(train_paths, label_map)
    test_ds = GeneExpressionDataset(test_paths, label_map)
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    num_cpus = os.cpu_count()
    num_workers = max(1, num_cpus // world_size - 1)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                            num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler,
                            num_workers=num_workers, pin_memory=True)


    # Load and wrap model with DDP
    model = FineTuneClassifier(PRETRAINED_CKPT, unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_model(model, train_loader, test_loader, optimizer, criterion, device, epoch, rank, wandb_enabled)

    # Save model only from rank 0
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        final_ckpt_path = os.path.join(SAVE_DIR, "finetuned_classifier.ckpt")
        torch.save(model.module.state_dict(), final_ckpt_path)
        torch.save(model.module.encoder.state_dict(), os.path.join(SAVE_DIR, "finetuned_encoder.ckpt"))

        if wandb_enabled:
            artifact = wandb.Artifact("finetuned_classifier", type="model")
            artifact.add_file(final_ckpt_path)
            wandb.log_artifact(artifact)

    dist.destroy_process_group()

# ===== SCRIPT ENTRY POINT =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(run_ddp, args=(world_size, args), nprocs=world_size, join=True)
