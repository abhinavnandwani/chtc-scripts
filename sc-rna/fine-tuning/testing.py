import os
import pickle
import random
import torch
import numpy as np
from scipy.sparse import load_npz
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch import nn
from tqdm import tqdm
from collections import Counter
from load import load_model_frommmf, gatherData

# ===== CONFIG =====
LABEL_PKL_PATH      = "../../c02x_split_seed42.pkl"
PRETRAINED_CKPT     = "./models/models.ckpt"
FINETUNED_CKPT      = "./models/finetuned_c02x/finetuned_epoch_1_preserved.ckpt"
DATA_DIR            = "../../MSSM"
VAL_BATCH_SIZE      = 1
SAMPLES_PER_DONOR   = 3
SEED                = 42

random.seed(SEED)

# ===== DATASET =====
class GeneExpressionDataset(Dataset):
    def __init__(self, file_paths, label_map, samples_per_donor=3):
        self.samples = []
        self.memory = {}
        for path in file_paths:
            donor_id = os.path.basename(path).replace("_aligned.npz", "")
            label = label_map.get(donor_id)
            if label is None:
                continue
            matrix = load_npz(path)
            self.memory[donor_id] = matrix
            k = min(samples_per_donor, matrix.shape[0])
            for i in random.sample(range(matrix.shape[0]), k):
                self.samples.append((donor_id, i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        donor_id, row_idx, label = self.samples[idx]
        row = self.memory[donor_id].getrow(row_idx).toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ===== MODEL WRAPPER =====
class FineTuneClassifier(nn.Module):
    def __init__(self, pretrained_ckpt, finetuned_ckpt, n_classes=1):
        super().__init__()
        # Load pretrained model
        self.model, self.model_config = load_model_frommmf(pretrained_ckpt)
        self.token_emb = self.model.token_emb
        self.pos_emb = self.model.pos_emb
        self.encoder = self.model.encoder

        self.norm = nn.LayerNorm(self.model_config['encoder']['hidden_dim'], elementwise_affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_config['encoder']['hidden_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        # Load fine-tuned weights (only the relevant parts)
        finetuned_ckpt_data = torch.load(finetuned_ckpt, map_location='cpu')
        finetuned_state_dict = finetuned_ckpt_data['gene']['state_dict']

        own_state = self.state_dict()
        loaded_keys = 0
        for name, param in finetuned_state_dict.items():
            full_name = f"model.{name}" if name.startswith("encoder") or name.startswith("token_emb") or name.startswith("pos_emb") else name
            if full_name in own_state:
                own_state[full_name].copy_(param)
                loaded_keys += 1
        print(f"[INFO] Loaded {loaded_keys} matching keys from finetuned checkpoint.")

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
        x = self.norm(x)
        return self.classifier(x)

# ===== MAIN =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label split
    with open(LABEL_PKL_PATH, 'rb') as f:
        label_data = pickle.load(f)
    test_ids = set(label_data['test']['SubID'])
    label_map = dict(zip(label_data['test']['SubID'], label_data['test']['c02x']))

    donor_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_aligned.npz")]
    donor_ids = [os.path.splitext(f.replace("_aligned", ""))[0] for f in donor_files]
    test_paths = [
        os.path.join(DATA_DIR, f)
        for f, did in zip(donor_files, donor_ids)
        if did in test_ids
    ]

    print(f"[INFO] Using {len(test_paths)} donors, sampling {SAMPLES_PER_DONOR} cells per donor.")
    if not test_paths:
        raise ValueError("No test donors found.")

    test_ds = GeneExpressionDataset(test_paths, label_map, samples_per_donor=SAMPLES_PER_DONOR)
    test_loader = DataLoader(test_ds, batch_size=VAL_BATCH_SIZE, num_workers=os.cpu_count())

    # ===== Load model with updated weights =====
    model = FineTuneClassifier(PRETRAINED_CKPT, FINETUNED_CKPT).to(device)
    model.eval()
    print("[INFO] Model loaded with pretrained base + finetuned weights.")

    # ===== Evaluation loop =====
    preds, truths = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model({'x': x})
            if torch.isnan(logits).any():
                print("[ERROR] Logits contain NaNs")

            preds.extend([int(p[0]) for p in (logits > 0).long().cpu().tolist()])
            truths.extend([int(t[0]) for t in y.cpu().long().tolist()])

    acc = balanced_accuracy_score(truths, preds)
    print(f"\n[RESULT] Balanced Accuracy on {len(test_ds)} samples: {acc:.4f}")
    print("Prediction counts:", Counter(preds))
    print("Ground truth counts:", Counter(truths))
    print("Confusion matrix:")
    print(confusion_matrix(truths, preds))

if __name__ == '__main__':
    main()
