import scanpy as sc
from sklearn.preprocessing import StandardScaler
import numpy as np
import anndata as ad

import matplotlib.pyplot as plt

def plot_embeddings_umap(
    embeddings: np.ndarray, 
    embs_key: str, 
    color_key: str, 
    title: str, 
    axes: plt.Axes = None) -> None:
    """
    Plots UMAP embeddings with specified color coding.

    Parameters:
    - embeddings (np.ndarray): The embeddings to plot.
    - embs_key (str): The key to store embeddings in adata.obsm.
    - color_key (str): The key for coloring the UMAP plot.
    - title (str): The title of the plot.
    - axes (plt.Axes, optional): The axes to plot on. If None, new axes are created.
    """
    
    if axes is None:
    _, axes = plt.subplots(figsize=(8, 8))

    scaler = StandardScaler()

    # Store the "before" embeddings in adata.obsm
    adata.obsm[embs_key] = embeddings
    # Compute neighbors and UMAP based on these UNTRAINED embeddings
    sc.pp.neighbors(adata, use_rep=embs_key)
    sc.tl.umap(adata, min_dist=0.5, random_state=42)

    adata.obsm["X_umap"] = scaler.fit_transform(adata.obsm["X_umap"])

    # Plot the result, but COLOR by the FINAL cluster assignments
    sc.pl.umap(
    adata,
    color=color_key,  # Use final clusters for coloring
    ax=axes,
    show=False,
    title=title,
    palette="tab20",
    legend_loc=None,
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim([-2.5, 2.5])
    axes.set_ylim([-2.5, 2.5])

plot_embeddings_umap(embeddings_before, 
             embs_key="GCN_before_training", 
             color_key="GCN_leiden_clusters", 
             title="Before Training (Epoch 1)", 
             axes=axes[0]
             )

plot_embeddings_umap(embeddings_after, 
             embs_key="GCN_after_training", 
             color_key="GCN_leiden_clusters", 
             title="After Training (Epoch 1)", 
             axes=axes[1]
             )
