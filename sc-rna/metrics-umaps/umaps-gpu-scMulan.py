import os
import sys
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# === Parameters ===
meta_path = "MSSM_meta_obs.csv"
output_dir = "./"
embedding_key = "X"  # use_rep key
n_jobs = multiprocessing.cpu_count()  # Use all CPU cores

method_paths = {
    # "scGPT": "scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body/",
    # "scMulan": "scMulan-main/results/zero-shot/extracted_cell_embeddings/",
    "UCE": "UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings/"
    # "scFoundation": "scFoundation/extracted_cell_embeddings/",
    # "Geneformer": "Geneformer_30M/extracted_cell_embeddings/"
}

# === Load metadata ===
meta = pd.read_csv(meta_path)
meta.index = meta.index.astype(str)

# === Subsample first 20 donors ===
donors = meta["SubID"].unique()[:20]
meta = meta[meta["SubID"].isin(donors)]

# === Process each method ===
for method, path in tqdm(method_paths.items(), desc="Processing methods", file=sys.stdout):
    tqdm.write(f"Processing UMAP for method: {method}", file=sys.stdout)

    # === Load donor-wise embeddings ===
    cell_list = []
    for donor in tqdm(donors, desc=f"Loading donor files for {method}", file=sys.stdout):
        donor_file = os.path.join(path, f"{donor}.csv")
        if os.path.exists(donor_file):
            df = pd.read_csv(donor_file, index_col=0)
            cell_list.append(df)
        else:
            tqdm.write(f"Missing file: {donor_file}", file=sys.stdout)

    if not cell_list:
        raise RuntimeError(f"No donor files found for method: {method}")

    # === Merge & Align ===
    tqdm.write("Concatenating donor data", file=sys.stdout)
    cells = pd.concat(cell_list, axis=0)

    if cells.shape[0] != meta.shape[0]:
        raise ValueError(f"Mismatch in cell count: cells={cells.shape[0]} vs meta={meta.shape[0]}")

    cells.index = meta.index
    adata = ad.AnnData(X=cells, obs=meta, var=cells.columns.to_frame())

    # === Compute neighbors ===
    tqdm.write("Computing nearest neighbors", file=sys.stdout)
    sc.pp.neighbors(adata, use_rep=embedding_key)

    # === Run UMAP ===
    tqdm.write(f"Running UMAP using {n_jobs} threads", file=sys.stdout)
    sc.tl.umap(adata, min_dist=0.3, n_components=2, n_jobs=n_jobs)

    # === Plot & Save ===
    tqdm.write("Generating UMAP plot", file=sys.stdout)
    sc.pl.umap(
        adata,
        color=["class", "subclass"],
        frameon=False,
        wspace=0.4,
        size=0.8,
        title=[f"{method}: class", f"{method}: subclass"]
    )
    out_path = os.path.join(output_dir, f"{method}_umap_cpu.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    tqdm.write(f"UMAP plot saved to: {out_path}", file=sys.stdout)

    # === Optional: Save AnnData object ===
    adata_path = os.path.join(output_dir, f"{method}_umap.h5ad")
    adata.write(adata_path)
    tqdm.write(f"AnnData object saved to: {adata_path}", file=sys.stdout)

    # === Cleanup ===
    del adata, cells, cell_list
