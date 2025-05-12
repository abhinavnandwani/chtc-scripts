import os
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rapids_singlecell import get, pp as rpp, tl as rtl

# # === GPU memory setup ===
# rmm.reinitialize(pool_allocator=True, initial_pool_size="72GiB", maximum_pool_size="78GiB")
# cp.cuda.set_allocator(rmm_cupy_allocator)

# === GPU memory setup ===
rmm.reinitialize(pool_allocator=True)
cp.cuda.set_allocator(rmm_cupy_allocator)

# === Parameters ===
meta_path = "MSSM_meta_obs.csv"
output_dir = "./"
embedding_key = "X"
cluster_key = "leiden"
label_key = "subclass"
use_rep_x = True

method_paths = {
    # "scGPT": "scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body/",
     "scMulan": "scMulan-main/results/zero-shot/extracted_cell_embeddings/",
    #"UCE": "UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings/"
    # "scFoundation": "scFoundation/extracted_cell_embeddings/"
    # "Geneformer": "Geneformer_30M/extracted_cell_embeddings/"
}

# === Output CSV setup ===
metrics_output = os.path.join(output_dir, "scMulan_metrics.csv")
columns = ["Model", "ASW", "ARI", "NMI", "AvgBIO"]
if not os.path.exists(metrics_output):
    pd.DataFrame(columns=columns).to_csv(metrics_output, index=False)

# === Load metadata ===
meta = pd.read_csv(meta_path)
meta.index = meta.index.astype(str)
donors = meta["SubID"].unique()

# === Main loop ===
for method, path in method_paths.items():
    print(f"\nComputing metrics for method: {method}")
    cp.get_default_memory_pool().free_all_blocks()

    cell_list = []
    for donor in tqdm(donors, desc=f"Loading {method}"):
        donor_file = os.path.join(path, f"{donor}.csv")
        if os.path.exists(donor_file):
            df = pd.read_csv(donor_file, index_col=0)
            cell_list.append(df)

    if not cell_list:
        raise RuntimeError(f"No CSVs found for method: {method}")

    cells = pd.concat(cell_list, axis=0)
    if cells.shape[0] != meta.shape[0]:
        raise ValueError(f"Row mismatch: cells={cells.shape[0]} vs meta={meta.shape[0]}")

    cells.columns = cells.columns.astype(str)
    cells.index = meta.index

    # === AnnData creation ===
    adata = ad.AnnData(X=cells, obs=meta, var=cells.columns.to_frame())
    get.anndata_to_GPU(adata)

    # === RAPIDS clustering ===
    rpp.neighbors(adata, use_rep=embedding_key)
    rtl.leiden(adata, resolution=0.2, key_added=cluster_key)

    # === Metrics computation ===
    print("Computing metrics...")
    X_gpu = cp.asarray(adata.X.to_numpy() if hasattr(adata.X, 'to_numpy') else adata.X)
    true_labels = adata.obs[label_key].astype("category").cat.codes.to_numpy()
    pred_labels = adata.obs[cluster_key].astype(int).to_numpy()

    asw = float(cython_silhouette_score(X_gpu, cp.asarray(pred_labels), metric="euclidean", chunksize=10000))
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
    avgBIO = (asw + ari + nmi) / 3

    pd.DataFrame([[method, asw, ari, nmi, avgBIO]], columns=columns)\
        .to_csv(metrics_output, mode='a', header=False, index=False)

    print(f"✓ Saved metrics to: {metrics_output}")

    # Clean up
    del adata, cells, cell_list
    cp.get_default_memory_pool().free_all_blocks()
