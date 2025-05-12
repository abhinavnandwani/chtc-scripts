import os
import pandas as pd
import scanpy as sc
import anndata
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

# RAPIDS & sklearn metrics
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# RAPIDS single-cell utilities
from rapids_singlecell import pp as rpp
from rapids_singlecell import tl as rtl

# === GPU memory setup ===
rmm.reinitialize(pool_allocator=True)
cp.cuda.set_allocator(rmm_cupy_allocator)

# === Parameters ===
input_dir = "scMulan/cell_embeddings/"
output_file = "scMulan-metrics.csv"
embedding_key = "X_scMulan"
cluster_key = "leiden"
label_key = "subclass"
num_partitions = 80  # Load first N files


# === CSV Output Setup ===
columns = ["Model", "ASW", "ARI", "NMI", "AvgBIO"]
if not os.path.exists(output_file):
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

# === Load AnnData files ===
adatas = []
for i in range(1, num_partitions + 1):
    file_path = os.path.join(input_dir, f"partition_{i}.h5ad")
    if os.path.exists(file_path):
        print(f"Loading partition {i}")
        adata = sc.read_h5ad(file_path)
        adata.obs["partition_id"] = i
        adatas.append(adata)
    else:
        print(f"Warning: {file_path} not found!")

if not adatas:
    raise RuntimeError("No .h5ad partitions found.")

# === Merge and subset ===
adata = anndata.concat(adatas, join="inner", label="batch", keys=None, index_unique=None)


# === Check embedding ===
if embedding_key not in adata.obsm:
    raise KeyError(f"Embedding '{embedding_key}' not found in .obsm.")

# === RAPIDS clustering ===
print("Running neighbors + Leiden clustering...")
rpp.neighbors(adata, use_rep=embedding_key)
rtl.leiden(adata, resolution=0.2)

# === Extract arrays ===
X_cpu = adata.obsm[embedding_key]
X_gpu = cp.asarray(X_cpu)

true_labels = adata.obs[label_key].astype("category").cat.codes.to_numpy()
pred_labels = adata.obs[cluster_key].astype(int).to_numpy()

# === Metrics ===
print("Computing metrics...")

asw_gpu = float(cython_silhouette_score(X_gpu, cp.asarray(pred_labels), metric="euclidean",chunksize=10000))
ari = float(adjusted_rand_score(true_labels, pred_labels))
nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
avgBIO = (asw_gpu + ari + nmi) / 3

# === Save ===
pd.DataFrame([["scMulan", asw_gpu , ari, nmi, avgBIO]], columns=columns)\
  .to_csv(output_file, mode='a', header=False, index=False)

print(f"Done! Metrics saved to: {output_file}")
