import os

import gseapy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def get_control_and_test_profiles(
    pert_id_to_test, dataset, df_gene_mapping, model, device
):
    """
    Get the control and test profiles for a given perturbation ID.

    Args:
        pert_id_to_test (str): The perturbation ID to test.
        dataset (LincsDataset): The dataset containing the profiles.
        pert_id_control (str, optional): The control perturbation ID. Defaults to "DMSO".

    Returns:
        tuple: Condition and expression tensors for the test and control profiles.
    """

    condition_x, expression_x, _ = dataset.get_item_by_pert_id(pert_id_to_test)

    control_profile = dataset.df_expression.mean()

    all_gene_names = control_profile.index.tolist()
    all_gene_names = (
        df_gene_mapping.loc[control_profile.index.astype(int)].values.squeeze().tolist()
    )
    control_x = control_profile.values

    condition_x = condition_x.unsqueeze(0).to(device)
    expression_x = expression_x.unsqueeze(0).to(device)

    recon_x, _, _ = model(expression_x, condition_x)

    return expression_x, condition_x, recon_x, control_x, all_gene_names


def get_monte_carlo_generation(
    condition_x: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 1000,
) -> torch.Tensor:
    """
    Generate gene expression profiles using Monte Carlo sampling from the model's latent space.

    Args:
        condition_x (torch.Tensor): Condition tensor for molecular embedding.
        model (torch.nn.Module): The trained model with decoder and molecular_embedding.
        device (torch.device): Device to run the computation on.
        num_samples (int, optional): Number of Monte Carlo samples to generate. Defaults to 1000.

    Returns:
        torch.Tensor: Mean generated gene expression profile across all samples.
    """
    all_generated_profiles = []
    for i in range(num_samples):
        torch.manual_seed(i)
        # Generate a new random latent vector
        z = torch.randn(condition_x.shape[0], model.latent_dim).to(device)
        c_embs = model.molecular_embedding(condition_x)

        z = torch.cat((z, c_embs), dim=1)

        generated_x = model.decoder(z)

        all_generated_profiles.append(generated_x.detach().cpu())

    mean_generated_x = torch.mean(torch.stack(all_generated_profiles), dim=0)

    return mean_generated_x


def calculate_log2_fold_change_from_control(recon_x, control_np, all_gene_names):
    """Calculate log2 fold change from the control profile.

    Args:
        recon_x (torch.Tensor): The reconstructed gene expression profile.
        control_np (np.ndarray): The control gene expression profile.
        all_gene_names (list): List of gene names corresponding to the expression profiles.
    """
    recon_x_np = recon_x.detach().cpu().numpy()
    log2_fold_change = recon_x_np - control_np

    ranked_gene_df = pd.DataFrame(
        {"gene_name": all_gene_names, "log2fc": log2_fold_change.squeeze()}
    )

    return ranked_gene_df.sort_values(by="log2fc", ascending=False)


def get_gsea_prerank(
    recon_x: torch.Tensor, control_np: np.ndarray, all_gene_names: list[str]
) -> pd.DataFrame:
    """
    Perform GSEA prerank analysis using reconstructed gene expression and control profile.

    Args:
        recon_x (torch.Tensor): The reconstructed gene expression profile.
        control_np (np.ndarray): The control gene expression profile.
        all_gene_names (list[str]): List of gene names corresponding to the expression profiles.

    Returns:
        pd.DataFrame: DataFrame containing significant GSEA results (FDR q-val < 0.05).
    """
    ranked_gene_df = calculate_log2_fold_change_from_control(
        recon_x, control_np, all_gene_names
    )
    prerank_obj = gseapy.prerank(
        rnk=ranked_gene_df, gene_sets="KEGG_2021_Human", seed=42
    )

    results_df = prerank_obj.res2d

    ranked_gene_df = ranked_gene_df.set_index("gene_name")
    significant_results = results_df[results_df["FDR q-val"] < 0.05]

    return significant_results


def evaluate_recon_and_gen_gsea_for_pert(
    pert_id_to_test: str,
    drug_name: str,
    val_dataset,
    df_gene_mapping: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    img_save_path: str = None,
) -> tuple[float, float]:
    """
    Evaluate GSEA results for original, reconstructed, and generated gene expression profiles for a given perturbation.

    Args:
        pert_id_to_test (str): Perturbation ID to test.
        drug_name (str): Name of the drug for labeling plots.
        val_dataset: Validation dataset containing gene expression profiles.
        df_gene_mapping (pd.DataFrame): DataFrame mapping gene IDs to gene names.
        model (torch.nn.Module): Trained model for reconstruction and generation.
        device (torch.device): Device to run computations on.
        img_save_path (str, optional): Path to save the GSEA comparison plot. Defaults to None.

    Returns:
        tuple[float, float]: GSEA reconstruction error and GSEA generation error.
    """
    expression_x, condition_x, recon_x, control_x, all_gene_names = (
        get_control_and_test_profiles(
            pert_id_to_test, val_dataset, df_gene_mapping, model, device
        )
    )

    mean_generated_x = get_monte_carlo_generation(condition_x, model, device)

    df_generated_results = get_gsea_prerank(mean_generated_x, control_x, all_gene_names)
    df_expression_results = get_gsea_prerank(expression_x, control_x, all_gene_names)
    df_reconstructed_results = get_gsea_prerank(recon_x, control_x, all_gene_names)

    df_results = pd.concat(
        [
            df_expression_results.set_index("Term")["ES"],
            df_reconstructed_results.set_index("Term")["ES"],
            df_generated_results.set_index("Term")["ES"],
        ],
        axis=1,
        keys=[
            "Original Expression",
            "Reconstructed Expression",
            "Generated Expression",
        ],
    )

    df_results = df_results.dropna(subset="Original Expression")
    df_results = df_results.fillna(0)

    gsea_recon_error: float = (
        np.abs(
            df_results["Original Expression"] - df_results["Reconstructed Expression"]
        )
        .sum()
        .tolist()
    )
    gsea_gen_error: float = (
        np.abs(df_results["Original Expression"] - df_results["Generated Expression"])
        .sum()
        .tolist()
    )

    df_plot = pd.melt(
        df_results.dropna(subset="Original Expression").iloc[:10].reset_index(),
        id_vars=["Term"],
        var_name="Profile Type",
        value_name="Enrichment Score",
    )

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create the bar plot
    ax = sns.barplot(
        data=df_plot,
        y="Term",
        x="Enrichment Score",
        hue="Profile Type",
        orient="h",
    )

    # Add a vertical line at zero for reference
    ax.axvline(0, color="black", lw=1)
    ax.grid(False)
    ax.set_title(f"GSEA Comparison for {drug_name}", fontsize=16)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=12)
    ax.set_ylabel("")
    plt.legend(title="Profile Source")

    if img_save_path is not None:
        plt.savefig(os.path.join(img_save_path, f"gsea_comparison_{drug_name}.png"))

    return gsea_recon_error, gsea_gen_error


def get_recon_correlation(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    img_save_path: str,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute and plot the correlation between reconstructed and original gene expression profiles
    for both training and validation datasets.

    Args:
        model (torch.nn.Module): Trained model for reconstruction.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        img_save_path (str): Path to save the residuals plot.
        device (torch.device): Device to run computations on.

    Returns:
        tuple[float, float]: Correlation coefficients for training and validation data.
    """
    train_expression = []
    train_reconstruction = []
    model = model.to(device)

    for condition, expression, moa_label in train_loader:
        condition = condition.to(device)
        expression = expression.to(device)

        recon_x, mu, log_var = model(expression, condition)
        train_reconstruction.append(recon_x.detach().cpu().numpy())
        train_expression.append(expression.detach().cpu().numpy())

    train_expression = np.vstack(train_expression)
    train_reconstruction = np.vstack(train_reconstruction)

    val_expression = []
    val_reconstruction = []
    for condition, expression, moa_label in val_loader:
        condition = condition.to(device)
        expression = expression.to(device)

        recon_x, mu, log_var = model(expression, condition)

        val_reconstruction.append(recon_x.detach().cpu().numpy())
        val_expression.append(expression.detach().cpu().numpy())
    val_expression = np.vstack(val_expression)
    val_reconstruction = np.vstack(val_reconstruction)

    val_corr = np.corrcoef(val_reconstruction.flatten(), val_expression.flatten())
    train_corr = np.corrcoef(train_reconstruction.flatten(), train_expression.flatten())

    _, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    axes[0].hist(
        train_reconstruction.flatten() - train_expression.flatten(),
        bins=np.linspace(-3, 3, 20),
        alpha=0.5,
        label="Training Data",
    )
    axes[1].hist(
        val_reconstruction.flatten() - val_expression.flatten(),
        bins=np.linspace(-3, 3, 20),
        alpha=0.5,
        label="Validation Data",
    )
    axes[0].set_title(
        "Distribution of Reconstruction Residuals (Training Data)\nCorrelation: {:.2f}".format(
            train_corr[0, 1]
        )
    )
    axes[1].set_title(
        "Distribution of Reconstruction Residuals (Validation Data)\nCorrelation: {:.2f}".format(
            val_corr[0, 1]
        )
    )
    axes[0].set_xlabel("Reconstructed Residuals")
    axes[0].set_ylabel("Frequency")
    axes[1].set_xlabel("Reconstructed Residuals")
    plt.tight_layout()

    plt.savefig(os.path.join(img_save_path, "residuals.png"))

    return train_corr[0, 1], val_corr[0, 1]
