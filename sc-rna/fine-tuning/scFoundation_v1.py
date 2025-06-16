import os
import sys
import signal
import gc
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import load_npz
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

from load import load_model_frommmf

# ===== CONFIG =====
LABEL_PKL_PATH     = "../../c02x_split_seed42.pkl"
PRETRAINED_CKPT    = "./models/scFoundation_pretrained.ckpt"
SAVE_DIR           = "./models/finetuned_c02x"
UNFREEZE_LAST_N    = 2
EPOCHS             = 5
BATCH_SIZE         = 4
LEARNING_RATE      = 1e-4
RANDOM_SEED        = 42

model = None  # For cleanup on interrupt

def handle_interrupt(signal_received, frame):
    print("\n[INFO] Interrupt received. Cleaning up...")
    if model:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_partial_interrupt.ckpt"))
        print("[INFO] Partial checkpoint saved.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

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

        for param in self.parameters():
            param.requires_grad = False
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
        x = batch['x'].unsqueeze(2)
        B, N, _ = x.shape
        device = x.device

        encoder_position_gene_ids = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        padding_label = torch.zeros(B, N, dtype=torch.bool, device=device)

        output = self.model(
            x.squeeze(2),
            padding_label=padding_label,
            encoder_position_gene_ids=encoder_position_gene_ids,
            encoder_labels=torch.ones_like(padding_label),
            decoder_data=x.squeeze(2),
            decoder_data_padding_labels=padding_label,
            decoder_position_gene_ids=encoder_position_gene_ids,
            mask_gene_name=False,
            mask_labels=None,
            output_attentions=False
        )

        x = torch.max(output, dim=1).values
        x = self.norm(x)
        return self.classifier(x)

# ===== TRAINING =====
def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss, preds, truths = 0, [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model({'x': x})
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend((logits > 0).long().cpu().tolist())
            truths.extend(y.cpu().long().tolist())

        acc = balanced_accuracy_score(truths, preds)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader.dataset):.4f} | Balanced Acc: {acc:.4f}")

        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                logits = model({'x': x})
                val_preds.extend((logits > 0).long().cpu().tolist())
                val_truths.extend(y.cpu().long().tolist())
        val_acc = balanced_accuracy_score(val_truths, val_preds)
        print(f"             Val Balanced Acc: {val_acc:.4f}")

        ckpt_path = os.path.join(SAVE_DIR, f"finetuned_epoch_{epoch+1}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to donor *_aligned.npz files')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)
    label_map = dict(zip(label_data['train']['SubID'], label_data['train']['c02x']))

    donor_paths = [
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.endswith("_aligned.npz") and os.path.splitext(f.replace("_aligned", ""))[0] in label_map
    ]

    if not donor_paths:
        raise ValueError("No labeled aligned donor files found.")

    donor_ids = [os.path.splitext(os.path.basename(f).replace("_aligned", ""))[0] for f in donor_paths]
    donor_labels = [label_map[did] for did in donor_ids]

    train_paths, test_paths = train_test_split(
        donor_paths, test_size=0.2, stratify=donor_labels, random_state=RANDOM_SEED
    )

    train_ds = GeneExpressionDataset(train_paths, label_map)
    test_ds = GeneExpressionDataset(test_paths, label_map)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    global model
    model = FineTuneClassifier(PRETRAINED_CKPT, unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, test_loader, optimizer, criterion, device, EPOCHS)

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_classifier.ckpt"))
    torch.save(model.encoder.state_dict(), os.path.join(SAVE_DIR, "finetuned_encoder.ckpt"))
    print("[INFO] Final model saved.")

if __name__ == '__main__':
    main()
