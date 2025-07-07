import os
import sys
import gc
import pickle
import random
import numpy as np
from scipy.sparse import load_npz
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from load import load_model_frommmf, gatherData
import wandb
import re

# ===== CONFIG =====
LABEL_PKL_PATH      = "../../c02x_split_seed42.pkl"
PRETRAINED_CKPT     = "./models/models.ckpt"
SAVE_DIR            = "./models/finetuned_c02x"
RESUME_CKPT_PATH    = None  # Will be set dynamically if a run is given
WANDB_RUN_ID        = "visionary-bird-47"  # Set to None to start a new run
UNFREEZE_LAST_N     = 3
EPOCHS              = 3
BATCH_SIZE          = 30
LEARNING_RATE       = 1e-4
RANDOM_SEED         = 42
USE_FULL_DATASET    = True
NUM_SAMPLE_CELLS    = 1000
RATIO_LABEL_0       = 0.6

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

import atexit
atexit.register(cleanup)

class GeneExpressionDataset(Dataset):
    def __init__(self, file_paths, label_map):
        self.samples = []
        self.memory = {}
        self.label_0 = []
        self.label_1 = []
        random.seed(RANDOM_SEED)

        for path in file_paths:
            donor_id = os.path.basename(path).replace("_aligned.npz", "")
            label = label_map.get(donor_id)
            if label is None:
                continue

            matrix = load_npz(path)
            self.memory[donor_id] = matrix

            for i in range(matrix.shape[0]):
                sample = (donor_id, i, label)
                if label == 0:
                    self.label_0.append(sample)
                else:
                    self.label_1.append(sample)

        if USE_FULL_DATASET:
            self.samples = self.label_0 + self.label_1
        else:
            n_label_0 = int(NUM_SAMPLE_CELLS * RATIO_LABEL_0)
            n_label_1 = NUM_SAMPLE_CELLS - n_label_0

            if len(self.label_0) < n_label_0 or len(self.label_1) < n_label_1:
                raise ValueError("Not enough samples in one of the label groups for the requested split.")

            self.samples = random.sample(self.label_0, n_label_0) + random.sample(self.label_1, n_label_1)
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        donor_id, row_idx, label = self.samples[idx]
        row = self.memory[donor_id].getrow(row_idx).toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class FineTuneClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes=1, unfreeze_last_n=2, resume_ckpt_path=None):
        super().__init__()
        self.model, self.model_config = load_model_frommmf(ckpt_path)
        self.token_emb = self.model.token_emb
        self.pos_emb = self.model.pos_emb
        self.encoder = self.model.encoder

        for param in self.parameters():
            param.requires_grad = False
        for layer in self.encoder.transformer_encoder[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_config['encoder']['hidden_dim'], elementwise_affine=True),
            nn.Linear(self.model_config['encoder']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        if resume_ckpt_path:
            print(f"[INFO] Loading from resume checkpoint: {resume_ckpt_path}")
            ckpt = torch.load(resume_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('gene', {}).get('state_dict', {})
            own_state = self.state_dict()

            loaded, skipped = 0, 0
            for k, v in state_dict.items():
                k_local = k.replace("model.", "")
                if k_local in own_state:
                    own_state[k_local].copy_(v)
                    loaded += 1
                else:
                    skipped += 1
            print(f"[INFO] Resumed parameters: loaded={loaded}, skipped={skipped}")

    def forward(self, batch):
        x = batch['x']
        value_mask = x > 0
        input_data, padding_mask = gatherData(x, value_mask, pad_token_id=self.model_config['pad_token_id'])
        gene_ids = torch.arange(x.size(1), device=x.device).repeat(x.size(0), 1)
        pos_ids, _ = gatherData(gene_ids, value_mask, pad_token_id=self.model_config['pad_token_id'])

        x = self.token_emb(input_data.unsqueeze(-1).float(), output_weight=0)
        x += self.pos_emb(pos_ids)
        x = self.encoder(x, padding_mask)
        x = torch.max(x, dim=1).values
        return self.classifier(x)

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, truths = 0, [], []

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

        total_loss += loss.item() * x.size(0)
        preds.extend((logits > 0).long().cpu().tolist())
        truths.extend(y.cpu().long().tolist())
        avg_loss = total_loss / len(preds)
        train_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return avg_loss

def run_training_loop(model, train_loader, optimizer, criterion, device, epochs, wandb_enabled, start_epoch):
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"\n========== Epoch {epoch + 1} ==========")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": train_loss
            }, step=epoch + 1)

        original_ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu')
        updated_state_dict = {
            k if k.startswith("model.") else f"model.{k}": v
            for k, v in model.state_dict().items()
        }

        if 'gene' in original_ckpt:
            original_ckpt['gene']['state_dict'] = updated_state_dict
        else:
            raise ValueError("Expected 'gene' key in pretrained checkpoint.")

        save_path = os.path.join(SAVE_DIR, f"finetuned_epoch_{epoch + 1}_preserved.ckpt")
        torch.save(original_ckpt, save_path)
        print(f"[INFO] Saved checkpoint: {save_path}")

        if wandb_enabled:
            artifact = wandb.Artifact(f"model_epoch_{epoch + 1}", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to *_aligned.npz files')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        wandb.login(key="f04957e341167ac5452921a251b0921fedd3558b")
        wandb_config = dict(
            project="scFoundation_finetune",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "unfreeze_last_n": UNFREEZE_LAST_N
            },
            mode="online"
        )
        if WANDB_RUN_ID:
            wandb_config.update({
                "id": WANDB_RUN_ID,
                "resume": "allow",
                "name": WANDB_RUN_ID
            })
        wandb.login()
        wandb.init(**wandb_config)
        wandb_enabled = True
    except Exception as e:
        print(f"[WARN] W&B disabled: {e}")
        wandb_enabled = False

    start_epoch = 0
    global RESUME_CKPT_PATH
    if wandb_enabled and WANDB_RUN_ID:
        try:
            api = wandb.Api()
            run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{WANDB_RUN_ID}")
            artifacts = run.logged_artifacts()
            model_artifacts = [a for a in artifacts if a.type == "model" and "finetuned_epoch" in a.name]

            def extract_epoch(artifact):
                match = re.search(r"epoch_(\d+)", artifact.name)
                return int(match.group(1)) if match else -1

            if model_artifacts:
                latest_artifact = max(model_artifacts, key=extract_epoch)
                latest_epoch = extract_epoch(latest_artifact)
                artifact_dir = latest_artifact.download()
                artifact_ckpt_path = os.path.join(artifact_dir, f"finetuned_epoch_{latest_epoch}_preserved.ckpt")

                RESUME_CKPT_PATH = artifact_ckpt_path
                start_epoch = latest_epoch
                print(f"[INFO] Auto-resuming from artifact: {latest_artifact.name} (epoch {start_epoch})")
        except Exception as e:
            print(f"[WARN] Failed to fetch latest W&B artifact: {e}")

    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)
    train_ids = set(label_data['train']['SubID'])
    label_map = dict(zip(label_data['train']['SubID'], label_data['train']['c02x']))

    donor_files = [f for f in os.listdir(args.data_path) if f.endswith("_aligned.npz")]
    donor_ids = [os.path.splitext(f.replace("_aligned", ""))[0] for f in donor_files]
    train_paths = [os.path.join(args.data_path, f)
                   for f, did in zip(donor_files, donor_ids) if did in train_ids]

    train_paths = train_paths[:16]
    print(f"[INFO] Using {len(train_paths)} donors.")

    if not train_paths:
        raise ValueError("No training donors found.")

    train_ds = GeneExpressionDataset(train_paths, label_map)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    global model
    model = FineTuneClassifier(PRETRAINED_CKPT, unfreeze_last_n=UNFREEZE_LAST_N, resume_ckpt_path=RESUME_CKPT_PATH).to(device)

    if wandb_enabled:
        wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    run_training_loop(model, train_loader, optimizer, criterion, device, EPOCHS, wandb_enabled, start_epoch)
    print("[INFO] Final model saved.")

if __name__ == '__main__':
    main()
