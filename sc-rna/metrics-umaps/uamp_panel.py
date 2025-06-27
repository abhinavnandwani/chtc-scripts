import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# === Parameters ===
COLOR_MODE = "class"
input_dir = "./umap_data"
output_dir = "./umap_output"
method1 = "scGPT"
method2 = "scMulan"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df1 = pd.read_csv(os.path.join(input_dir, f"{method1}_umap_data.csv"), index_col=0)
df2 = pd.read_csv(os.path.join(input_dir, f"{method2}_umap_data.csv"), index_col=0)
unique_labels = sorted(set(df1[COLOR_MODE]).union(df2[COLOR_MODE]))

# === Build extended palette ===
base = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"]  # Okabe-Ito

if len(unique_labels) > len(base):
    more = sns.color_palette("tab20", n_colors=len(unique_labels) - len(base))
    palette = base + [m for m in more]
else:
    palette = base[:len(unique_labels)]

label_to_color = dict(zip(unique_labels, palette))

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [1,1,0.7]}, dpi=300)

for ax, df, method in zip(axs[:2], [df1, df2], [method1, method2]):
    for lbl in unique_labels:
        sub = df[df[COLOR_MODE] == lbl]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=4,
                   color=label_to_color[lbl], alpha=0.6, linewidths=0)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(method); ax.set_aspect("equal"); ax.grid(False)

# Legend in 3rd column
axs[2].axis("off")
handles = [mpatches.Patch(color=label_to_color[lbl], label=lbl) for lbl in unique_labels]
axs[2].legend(handles=handles, loc="center", ncol=2, frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{method1}_{method2}_okabeito_palette.png"), dpi=300)
