import os
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from tqdm import tqdm

# === RAPIDS Setup ===
import rapids_singlecell as rsc
import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

# Initialize GPU memory
rmm.reinitialize(pool_allocator=True)
cp.cuda.set_allocator(rmm_cupy_allocator)

# === Parameters ===
meta_path = "MSSM_meta_obs.csv"
use_rep_x = True  # Set to False to use PCA
output_dir = "./"

method_paths = {
    #"scGPT": "scGPT-main/results/zero-shot/extracted_cell_embeddings_full_body/",
    #"scMulan": "scMulan-main/results/zero-shot/extracted_cell_embeddings/"
    # "UCE": "UCE_venv/UCE-main/results/cell_embeddings/extracted_cell_embeddings/",
     "scFoundation": "scFoundation/extracted_cell_embeddings/"
    # "Geneformer": "Geneformer_30M/extracted_cell_embeddings/"
}

# === Load Metadata ===
meta = pd.read_csv(meta_path)
meta.index = meta.index.astype(str)
donors = meta["SubID"].unique()

# === Iterate Over Methods ===
for method, path in method_paths.items():
    print(f"\n🚀 Processing {method}...")

    cells = pd.DataFrame()
    for donor in tqdm(donors, desc=f"📦 Loading {method}"):
        donor_file = os.path.join(path, f"{donor}.csv")
        if os.path.exists(donor_file):
            df = pd.read_csv(donor_file, index_col=0)
            cells = pd.concat([cells, df])
        else:
            print(f"⚠️ Warning: {donor_file} not found.")

    # === Sanity Check ===
    if cells.shape[0] != meta.shape[0]:
        raise ValueError(f"Mismatch in rows: cells={cells.shape[0]} vs meta={meta.shape[0]}")

    cells.columns = cells.columns.astype(str)
    cells.index = meta.index.astype(str)
    meta.index = meta.index.astype(str)

    # === Create AnnData ===
    adata = ad.AnnData(X=cells, obs=meta, var=cells.columns.to_frame())

    # === Move to GPU (in place) ===
    rsc.get.anndata_to_GPU(adata)          # ← no reassignment

    # === RAPIDS neighbors + UMAP ===
    if use_rep_x:
        rsc.pp.neighbors(adata, use_rep='X')
    else:
        rsc.pp.pca(adata, n_comps=100, use_highly_variable=False)
        rsc.pp.neighbors(adata)

    rsc.tl.umap(adata, min_dist=0.3)

    # === Back to CPU for visualization (in place) ===
    rsc.get.anndata_to_CPU(adata)          # ← no reassignment

    # === Plot and Save ===
    sc.pl.umap(
        adata,
        color=["class", "subclass"],
        frameon=False,
        wspace=0.4,
        size=0.8,
        title=[f"{method}: class", f"{method}: subclass"]
    )
    plt.savefig(os.path.join(output_dir, f"{method}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # === Cleanup ===
    del cells
    del adata
    cp.get_default_memory_pool().free_all_blocks()
