import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class LogisticRegressionModel(nn.Module):
    """
    Simple logistic regression:
    linear layer mapping input_dim -> num_classes
    """
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
def get_model_data_path(model_name):
    data_path="."

    if model_name=="scGPT":
        data_path="/media/sayalialatkar/T9/Sayali/FoundationModels/scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body"
    elif model_name=="UCE":
        data_path="/media/sayalialatkar/T9/Sayali/FoundationModels/UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings"
    elif model_name=="scFoundation":
        data_path="extracted_cell_embeddings_PASCODE_c90x"
    elif model_name=="Geneformer":
        data_path="/media/sayalialatkar/T9/Sayali/FoundationModels/Geneformer_30M/extracted_cell_embeddings"
    elif model_name=="scMulan":
        data_path="/media/sayalialatkar/T9/Sayali/FoundationModels/scMulan-main/results/zero-shot/extracted_cell_embeddings"

    return data_path


def get_metadata(meta_path, sample_column='SubID', phen_column='c15x',val_split=0.2, random_state=42):
    """
    Load metadata from data['train'] and return sample IDs, labels, and label map.
    """
    with open(meta_path, 'rb') as handle:
        data = pickle.load(handle)

    # Map phenotype labels
    # data['train'][phen_column] = data['train'][phen_column].map({"AD":1, "Control":0})
    # data['test'][phen_column] = data['test'][phen_column].map({"AD":1, "Control":0})

    df = data['train'].copy()

    sample_ids = df[sample_column].unique().tolist()
    labels = [df[df[sample_column] == sid][phen_column].iloc[0] for sid in sample_ids]

    train_ids, val_ids = train_test_split(
        sample_ids,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )

    label_map = dict(zip(df[sample_column], df[phen_column]))
    train_label_map = {sid: label_map[sid] for sid in train_ids}
    val_label_map = {sid: label_map[sid] for sid in val_ids}

    return train_ids, val_ids, train_label_map, val_label_map

def get_metadata_cv(meta_path, sample_column='SubID', phen_column='c15x'):
    """
    Load metadata from data['train'] and return sample IDs, labels, and label map.
    """
    with open(meta_path, 'rb') as handle:
        data = pickle.load(handle)

    df = data['train'].copy()
    # Map phenotype labels
    # df[phen_column] = df[phen_column].map({"AD":1, "Control":0})
    
    sample_ids = df[sample_column].unique().tolist()
    labels = [df.loc[df[sample_column] == sid, phen_column].iloc[0] for sid in sample_ids]
    label_map = dict(zip(df[sample_column], df[phen_column]))
    
    return sample_ids, labels, label_map


class EmbDataset(Dataset):
    def __init__(self, file_list, data_path, label_mapping,scaler=None, cache_data=True, file_format="csv"):
        self.file_list = file_list
        self.data_path = data_path
        self.file_format = file_format
        self.label_mapping = label_mapping
        self.index_mapping = []
        self.sample_weights = []
        self.cache_data = cache_data
        self.file_cache = {}
        self.scaler = scaler

        # Build index and weights
        for donor in file_list:
            file_path = os.path.join(data_path, donor + f".{self.file_format}")
            df = pd.read_csv(file_path, index_col=0) if self.file_format == "csv" else pd.read_parquet(file_path)
            n_cells = len(df)
            donor_weight = 1.0 / n_cells if n_cells > 0 else 0
            for i in range(n_cells):
                self.index_mapping.append((donor, i))
                self.sample_weights.append(donor_weight)
            if cache_data:
                self.file_cache[donor] = df.copy()
            del df

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        donor, local_idx = self.index_mapping[idx]
        if self.cache_data and donor in self.file_cache:
            df = self.file_cache[donor]
        else:
            file_path = os.path.join(self.data_path, donor + f".{self.file_format}")
            df = pd.read_csv(file_path, index_col=0) if self.file_format == "csv" else pd.read_parquet(file_path)
        row = df.iloc[local_idx]
        if self.scaler is not None:
            row = self.scaler.transform(row.values.reshape(1, -1)).squeeze(0)
        else:
            row=row.values
        data = torch.tensor(row, dtype=torch.float32)
        label = torch.tensor(int(self.label_mapping.get(donor, -1)), dtype=torch.float32)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return data, label, weight


