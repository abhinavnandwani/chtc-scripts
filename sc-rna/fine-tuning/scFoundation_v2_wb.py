import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import signal
import gc
import pickle
import numpy as np
from scipy.sparse import load_npz
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from load import *
import wandb
import pandas as pd
import atexit
from transformers import get_cosine_schedule_with_warmup

from load import load_model_frommmf

# ===== CONFIG =====
LABEL_PKL_PATH     = "../../c02x_split_seed42.pkl"
PRETRAINED_CKPT    = "./models/models.ckpt"
SAVE_DIR           = "./models/finetuned_c02x"
UNFREEZE_LAST_N    = 7
EPOCHS             = 25
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-4
RANDOM_SEED        = 42
CLASSIFIER_HIDDEN_DIMS = "512,128,64"  # comma-separated string


# === NEW: Subset Selection ===
N_CLASS0           = 10        # Number of donors with c02x = 0
N_CLASS1           = 10        # Number of donors with c02x = 1
SUBSET_LOG_CSV     = "subset_log.csv"

model = None

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


atexit.register(cleanup)


# ===== HELPER FUNCTIONS ====
def build_subset_log_dataframe(label_data, n_class0, n_class1, seed=42):
    """
    Select a balanced subset of training donors and return donor-level DataFrame.

    Args:
        label_data (dict): Dictionary containing 'train' key with 'SubID' and 'c02x' lists.
        n_class0 (int): Number of donors to sample with label 0.
        n_class1 (int): Number of donors to sample with label 1.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Donor-level DataFrame with columns: SubID, c02x, used (always True).
    """
    # Cell-level to donor-level
    donor_df = pd.DataFrame({
        "SubID": label_data['train']['SubID'],
        "c02x": label_data['train']['c02x']
    }).drop_duplicates(subset='SubID')

    # Sample donors per class
    class0 = donor_df[donor_df['c02x'] == 0].sample(n=n_class0, random_state=seed)
    class1 = donor_df[donor_df['c02x'] == 1].sample(n=n_class1, random_state=seed)

    selected_donors = pd.concat([class0, class1])
    selected_donors['used'] = True

    return selected_donors.reset_index(drop=True)

def build_classifier(input_dim, hidden_dims, output_dim):
    layers = [nn.LayerNorm(input_dim)]
    prev_dim = input_dim
    for hdim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hdim))
        #layers.append(nn.ReLU())
        layers.append(nn.LeakyReLU(negative_slope=0.01))  # changed cause dead neurons with ReLU
        prev_dim = hdim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

# ===== DATASET =====
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

# ===== MODEL =====
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
        # # Unfreeze last N transformer layers
        # for layer in self.encoder.transformer_encoder[-unfreeze_last_n:]:
        #     for param in layer.parameters():
        #         param.requires_grad = True

        # Unfreeze encoder according to policy
        if unfreeze_last_n == -1:
            print("[INFO] Unfreezing all encoder layers")
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            print(f"[INFO] Unfreezing last {unfreeze_last_n} encoder layers")
            for layer in self.encoder.transformer_encoder[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(self.model_config['encoder']['hidden_dim'], elementwise_affine=True),
        #     nn.Linear(self.model_config['encoder']['hidden_dim'], 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )
        hidden_dim = self.model_config['encoder']['hidden_dim']
        # Set classifier depth
        try:
            hidden_dims = list(map(int, CLASSIFIER_HIDDEN_DIMS.split(",")))
        except Exception as e:
            print(f"[WARN] Invalid CLASSIFIER_HIDDEN_DIMS format: {e} — using default [512, 128, 64]")
            hidden_dims = [512, 128, 64]
        self.classifier = build_classifier(hidden_dim, hidden_dims, n_classes)

    def forward(self, batch):
        x = batch['x']  # shape: (batch_size, num_genes)

        # Create boolean mask where expression > 0
        value_mask = x > 0

        # Use gatherData to get padded inputs and attention masks
        input_data, padding_mask = gatherData(x, value_mask, pad_token_id=self.model_config['pad_token_id'])

        # Prepare position IDs using the same value_mask
        gene_ids = torch.arange(x.size(1), device=x.device).repeat(x.size(0), 1)
        pos_ids, _ = gatherData(gene_ids, value_mask, pad_token_id=self.model_config['pad_token_id'])

        # Token + position embedding
        x = self.token_emb(input_data.unsqueeze(-1).float(), output_weight=0)
        x += self.pos_emb(pos_ids)

        # Forward through encoder
        x = self.encoder(x, padding_mask)  # NOTE: padding_mask is passed as 2nd positional arg

        # Pooling
        x = torch.max(x, dim=1).values
    
        return self.classifier(x)

    def get_param_groups(self):
        encoder_layers = self.encoder.transformer_encoder
        num_layers = len(encoder_layers)
        split_point = num_layers // 2

        # Always define top and base layer groups, even if some layers are frozen (they'll be filtered out)
        top_layer_params = [p for layer in encoder_layers[split_point:] for p in layer.parameters() if p.requires_grad]
        base_layer_params = [p for layer in encoder_layers[:split_point] for p in layer.parameters() if p.requires_grad]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]

        return [
            {"params": classifier_params, "lr": 1e-4},
            {"params": top_layer_params, "lr": 1e-4},
            {"params": base_layer_params, "lr": 5e-5},
        ]

#========= VAL SET EVAL =========#
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model({'x': x})
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            pred = (logits > 0).long()
            correct += (pred == y.long()).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ===== SINGLE EPOCH TRAINING =====
def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    preds, truths = [], []

    print("\n[TRAIN] Starting epoch...")
    train_pbar = tqdm(train_loader, desc="Training", unit="batch", leave=True,
                      dynamic_ncols=True, mininterval=0.1, smoothing=0.1, file=sys.stdout)

    for x, y in train_pbar:
        x, y = x.to(device), y.to(device).unsqueeze(1)

        logits = model({'x': x})
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)

        pred = (logits > 0).long()
        correct += (pred == y.long()).sum().item()
        total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        train_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})
        if wandb.run is not None:
            wandb.log({
                "batch/loss": loss.item(),
                "batch/acc": (pred == y.long()).float().mean().item(),
                "batch/lr": scheduler.get_last_lr()[0]
            })

    return avg_loss, acc


# ===== TRAINING LOOP =====
def run_training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, wandb_enabled):
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

        # ---- Train one epoch ----
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # ---- W&B Logging ----
        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": train_loss,
                "train/acc_epoch": train_acc
            })

        # ---- Validation ----
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if wandb_enabled:
            wandb.log({
                "val/loss_epoch": val_loss,
                "val/acc_epoch": val_acc
            })
        # ---- Cleanup ----
        torch.cuda.empty_cache()
        gc.collect()


        # ---- Load original ckpt, update only state_dict ----
        original_ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu')
        if 'gene' in original_ckpt:
            updated_state_dict = {
                k if k.startswith("model.") else f"model.{k}": v
                for k, v in model.state_dict().items()
            }
            original_ckpt['gene']['state_dict'] = updated_state_dict
        else:
            raise ValueError("Expected 'gene' key in pretrained checkpoint.")

        # ---- Save updated checkpoint ----
        ckpt_path = os.path.join(SAVE_DIR, f"finetuned_epoch_{epoch + 1}_preserved.ckpt")
        torch.save(original_ckpt, ckpt_path)
        print(f"[INFO] Saved updated checkpoint: {ckpt_path}")

        if wandb_enabled:
            artifact = wandb.Artifact(f"model_epoch_{epoch + 1}", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)





# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to donor *_aligned.npz files')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import wandb
        wandb.login(key="f04957e341167ac5452921a251b0921fedd3558b")
        wandb.init(
            project="scFoundation_finetune",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE
                "unfreeze_last_n": UNFREEZE_LAST_N
            },
            mode="online"
        )
        wandb_enabled = True
    except Exception as e:
        print(f"[WARN] W&B disabled: {e}")
        wandb_enabled = False

    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)

    train_df = pd.DataFrame({
        "SubID": label_data['train']['SubID'],
        "c02x": label_data['train']['c02x']
    })

    if N_CLASS0 == -1 and N_CLASS1 == -1:
        # Use all donors and log them
        train_df['used'] = True
        subset_log_df = train_df.copy()
        print(f"[INFO] Using all {len(train_df)} donors for training.")
    else:
        subset_log_df = build_subset_log_dataframe(label_data, N_CLASS0, N_CLASS1, seed=RANDOM_SEED)
        print(f"[INFO] Using subset with {len(subset_log_df)} donors.")
        

    # Save CSV log in both cases
    subset_log_df.to_csv(SUBSET_LOG_CSV, index=False)
    selected_train_ids = set(subset_log_df['SubID'])

    # Always log CSV to W&B
    if wandb_enabled:
        artifact = wandb.Artifact("subset_log", type="dataset")
        artifact.add_file(SUBSET_LOG_CSV)
        wandb.log_artifact(artifact)


    train_ids = selected_train_ids
    label_map = dict(zip(label_data['train']['SubID'], label_data['train']['c02x']))

    donor_files = [f for f in os.listdir(args.data_path) if f.endswith("_aligned.npz")]
    donor_ids = [os.path.splitext(f.replace("_aligned", ""))[0] for f in donor_files]

    train_paths = [
        os.path.join(args.data_path, f)
        for f, did in zip(donor_files, donor_ids)
        if did in train_ids
    ]

    # DEBUG: Restricting to only 1 donor for quick testing
    #train_paths = train_paths[:128]
    #print(f"[INFO] Using {len(train_paths)} train donors.")

    if not train_paths:
        raise ValueError("No training donors found.")

    # train_ds = GeneExpressionDataset(train_paths, label_map)
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    
    # total_cells = len(train_ds)
    # print(f"[INFO] Total number of cells: {total_cells}")

    ############# ADDED VAL SPLIT ################
    full_dataset = GeneExpressionDataset(train_paths, label_map)

    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))), test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )


    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())

    print(f"[INFO] Train cells: {len(train_ds)} | Val cells: {len(val_ds)}")
    ##############################################

    if wandb_enabled:
        wandb.config.train_cells = len(train_ds)
        wandb.config.val_cells = len(val_ds)
    global model
    
    model = FineTuneClassifier(
        PRETRAINED_CKPT,
        unfreeze_last_n=UNFREEZE_LAST_N
    ).to(device)

    if wandb_enabled:
        wandb.watch(model, log="all", log_freq=100)

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.get_param_groups())


    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)  # You can tune this ratio (10% is common)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    run_training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, EPOCHS, wandb_enabled)


    print("[INFO] Final model saved.")

if __name__ == '__main__':
    main()
