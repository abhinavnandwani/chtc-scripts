import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

from model.load import load_model_frommmf, gatherData  # scFoundation-specific

# -------------------------
# Configuration (EDIT these)
# -------------------------
LABEL_PKL_PATH     = "./labels/c02x_labels.pkl"             # Pickle file: donor-level labels for a specific contrast
PRETRAINED_CKPT    = "./checkpoints/scFoundation_pretrained.pt"  # Path to scFoundation checkpoint
SAVE_DIR           = "./outputs/finetuned_c02x"             # Where models will be saved
UNFREEZE_LAST_N    = 2         # Number of final transformer layers to unfreeze
EPOCHS             = 5         # Number of training epochs
BATCH_SIZE         = 64
LEARNING_RATE      = 1e-4
RANDOM_SEED        = 42

# -------------------------
# Dataset: One sample per cell, pulled from donor-level .npz
# -------------------------
class CSRGeneExpressionDataset(Dataset):
    def __init__(self, donor_paths, label_map):
        self.samples = []
        for path in donor_paths:
            donor_id = os.path.basename(path).split('.')[0]
            label = label_map.get(donor_id)
            if label is None:
                continue  # Skip donors not included in the contrast
            data = np.load(path)
            matrix = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
            for i in range(matrix.shape[0]):
                self.samples.append((path, i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, row_idx, label = self.samples[idx]
        data = np.load(path)
        matrix = csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
        row = matrix.getrow(row_idx).toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -------------------------
# Fine-tuning wrapper with classification head
# -------------------------
class FineTuneClassifier(nn.Module):
    def __init__(self, ckpt_path, n_classes=1, unfreeze_last_n=2):
        super().__init__()
        self.model, self.model_config = load_model_frommmf(ckpt_path)
        self.token_emb = self.model.token_emb
        self.pos_emb = self.model.pos_emb
        self.encoder = self.model.encoder

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze only the last N layers of the encoder (adaptation while avoiding overfitting)
        for layer in self.encoder.transformer_encoder[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Add classification head (output is a single logit)
        self.norm = nn.BatchNorm1d(self.model_config['encoder']['hidden_dim'], affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_config['encoder']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, batch):
        x = batch['x']
        value_labels = x > 0
        x, padding_mask = gatherData(x, value_labels, self.model_config['pad_token_id'])

        gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        pos_ids, _ = gatherData(gene_ids, value_labels, self.model_config['pad_token_id'])

        x = self.token_emb(x.unsqueeze(-1).float(), output_weight=0)
        x += self.pos_emb(pos_ids)

        x = self.encoder(x, padding_mask)
        x, _ = torch.max(x, dim=1)  # Global max pooling
        x = self.norm(x)
        return self.classifier(x)

# -------------------------
# Training loop (prints balanced accuracy)
# -------------------------
def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        preds, truths = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model({'x': x, 'targets': y})
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend((logits > 0).long().cpu().tolist())
            truths.extend(y.cpu().long().tolist())

        train_acc = balanced_accuracy_score(truths, preds)
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader.dataset):.4f} | Train Balanced Acc: {train_acc:.4f}")

        # Evaluation
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                logits = model({'x': x, 'targets': y})
                val_preds.extend((logits > 0).long().cpu().tolist())
                val_truths.extend(y.cpu().long().tolist())
        val_acc = balanced_accuracy_score(val_truths, val_preds)
        print(f"             Test Balanced Acc: {val_acc:.4f}")

# -------------------------
# Main entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to .npz files (1 per donor)')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load donor-level contrast labels
    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)
    label_map = dict(zip(label_data['train']['SubID'], label_data['train']['c02x']))  # Adjust key if needed

    # Only include donors that are labeled in the current contrast
    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    donor_paths = [os.path.join(args.data_dir, f) for f in all_files
                   if os.path.splitext(f)[0] in label_map]

    if len(donor_paths) == 0:
        raise ValueError("No labeled donor files found in data_dir.")

    # Prepare stratified split
    donor_ids = [os.path.splitext(os.path.basename(p))[0] for p in donor_paths]
    donor_labels = [label_map[did] for did in donor_ids]

    train_donors, test_donors = train_test_split(
        donor_paths,
        test_size=0.2,
        stratify=donor_labels,
        random_state=RANDOM_SEED
    )

    # Datasets and loaders
    train_ds = CSRGeneExpressionDataset(train_donors, label_map)
    test_ds = CSRGeneExpressionDataset(test_donors, label_map)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize and train model
    model = FineTuneClassifier(PRETRAINED_CKPT, unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, test_loader, optimizer, criterion, device, EPOCHS)

    # Save both versions of the model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_classifier.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(SAVE_DIR, "finetuned_encoder.pt"))
    print("Training complete. Models saved.")

# -------------------------
# Example usage:
# python finetune_scfoundation.py --data_dir ./data/donors_subset/
# -------------------------
if __name__ == '__main__':
    main()
