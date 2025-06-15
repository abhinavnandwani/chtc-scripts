from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

# Global setup (load once)
model_name = "scFoundation"
subclass_list = ["Oligo", "Micro", "Astro", "EN_L2_3_IT"]
phen_label = 'r01x'

FILE_PATH = f"scFoundation/extracted_cell_embeddings/"
BASE_SAVE_PATH = f"./results_scFoundation_pseudotime_r01x/"

clinical_meta = pd.read_csv("metadata_latest_oct5.csv", low_memory=False)
meta_obs = pd.read_csv("MSSM_meta_obs.csv")
meta_obs.index = meta_obs.barcodekey


def process_subclass(subclass):
    print(f"Processing subclass: {subclass}")

    SAVE_PATH = f"{BASE_SAVE_PATH}{phen_label}/"
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    phen_cell_meta_merge = pd.merge(
        meta_obs[['barcodekey', 'SubID', 'class', 'subclass']],
        clinical_meta[[phen_label, 'SubID']],
        on="SubID"
    )
    phen_cell_meta_merge.index = phen_cell_meta_merge.barcodekey
    phen_cell_meta_merge = phen_cell_meta_merge.dropna(subset=[phen_label])
    phen_cell_meta_merge_subset = phen_cell_meta_merge[phen_cell_meta_merge["subclass"] == subclass]
    final_donors = phen_cell_meta_merge_subset["SubID"].unique()

    adata = None

    for donor in final_donors:
        try:
            donor_df = pd.read_csv(f"{FILE_PATH}{donor}.csv", index_col=0)
            phen_meta_obs_donor = phen_cell_meta_merge_subset[phen_cell_meta_merge_subset["SubID"] == donor]
            donor_df.index = meta_obs[meta_obs["SubID"] == donor].index

            common_barcodes = list(set(donor_df.index).intersection(set(phen_meta_obs_donor.index)))
            phen_meta_obs_donor = phen_meta_obs_donor.loc[common_barcodes]
            donor_df = donor_df.loc[common_barcodes]

            if donor_df.empty or phen_meta_obs_donor.empty:
                continue

            temp_adata = ad.AnnData(
                donor_df.values,
                obs=phen_meta_obs_donor,
                var=donor_df.columns.to_frame()
            )

            if adata is None:
                adata = temp_adata
            else:
                adata = ad.concat([adata, temp_adata])
        except Exception as e:
            print(f"Skipping donor {donor} due to error: {e}")
            continue

    if adata is None or adata.shape[0] == 0:
        print(f"Skipping subclass {subclass} due to empty data.")
        return

    try:
        all_elements = np.flatnonzero(adata.obs[phen_label] == 0)
        if len(all_elements) == 0:
            print(f"No root found for subclass {subclass}")
            return

        root = all_elements[random.randrange(len(all_elements))]

        sc.pp.pca(adata)
        sc.pp.neighbors(adata, use_rep='X_pca')
        sc.tl.diffmap(adata)

        adata.uns['iroot'] = root
        sc.tl.dpt(adata)
        sc.tl.umap(adata)

        with plt.rc_context():
            sc.pl.umap(
                adata,
                color=['dpt_pseudotime', phen_label],
                show=False,
                frameon=False,
                color_map="magma"
            )
            plt.savefig(f"{SAVE_PATH}{model_name}_{phen_label}_{subclass}_pseudotime.png", dpi=300, bbox_inches="tight")

        print(f"Completed subclass {subclass}")
    except Exception as e:
        print(f"Error during pseudotime calculation for subclass {subclass}: {e}")


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=len(subclass_list)) as executor:
        futures = [executor.submit(process_subclass, subclass) for subclass in subclass_list]
        for future in futures:
            future.result()  # Raise any errors
