import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes.optim as bnb
import imageio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import yaml

from .dataset import LincsDataset
from .eval import evaluate_recon_and_gen_gsea_for_pert, get_recon_correlation
from .model import CVAE

ROOT = Path(os.getcwd()).parents[1]


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    latent_dim: int,
    beta: float = 1.0,
    recon_weight: float = 1000,
    free_bits_per_dim: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the loss function for a Variational Autoencoder (VAE), combining reconstruction loss and KL divergence.

    Args:
        recon_x (torch.Tensor): The reconstructed output from the decoder.
        x (torch.Tensor): The original input data.
        mu (torch.Tensor): The mean of the latent Gaussian distribution.
        log_var (torch.Tensor): The log variance of the latent Gaussian distribution.
        latent_dim (int): The dimensionality of the latent space.
        beta (float, optional): Weight for the KL divergence term. Default is 1.0.
        recon_weight (float, optional): Weight for the reconstruction loss term. Default is 1000.
        free_bits_per_dim (float, optional): Minimum KL divergence per latent dimension before penalizing (free bits). Default is 0.02.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Total VAE loss (torch.Tensor)
            - Reconstruction loss (torch.Tensor)
            - KL divergence (torch.Tensor)
    """
    # Reconstruction Loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Normalize KL divergence by batch size
    kl_div /= x.size(0)

    # Target 0.02 bits of information per latent dimension before penalizing
    kl_div = torch.clamp(kl_div, min=free_bits_per_dim * latent_dim)

    return recon_loss * recon_weight + beta * kl_div, recon_loss, kl_div


def plot_training_history(history: Dict[str, List[float]]):
    """
    Plot the training history of the model.

    Args:
        history (Dict[str, List[float]]): A dictionary containing lists of training and validation metrics.
            Expected keys: 'train_loss', 'val_loss', 'train_recon', 'val_recon',
            'train_kl', 'val_kl', 'train_class_loss', 'val_class_loss'.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plots.
    """
    _, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=False)
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")

    axes[1, 0].plot(history["train_recon"], label="Train Reconstruction Loss")
    axes[1, 0].plot(history["val_recon"], label="Val Reconstruction Loss")

    axes[1, 1].plot(history["train_kl"], label="Train KL Divergence")
    axes[1, 1].plot(history["val_kl"], label="Val KL Divergence")

    axes[0, 1].plot(history["train_class_loss"], label="Train Classification Loss")
    axes[0, 1].plot(history["val_class_loss"], label="Val Classification Loss")

    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[1, 0].set_title("Reconstruction Loss")
    axes[1, 0].legend()

    axes[1, 1].set_title("KL Divergence")
    axes[1, 1].legend()

    axes[0, 1].set_title("Classification Loss")
    axes[0, 1].legend()

    return axes


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: Dict[str, Any],
    beta: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    """
    Runs a single epoch of training or evaluation for a VAE-based model with optional classification.

    Args:
        model (nn.Module): The model to train or evaluate.
        loader (DataLoader): DataLoader providing batches of (condition, expression, moa_label).
        config (Dict[str, Any]): Configuration dictionary containing loss weights (e.g., 'alpha', 'gamma').
        beta (float): Weight for the KL divergence term in the VAE loss.
        device (torch.device): Device to run computations on (e.g., 'cpu' or 'cuda').
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None, runs in evaluation mode.

    Returns:
        Dict[str, float]: Dictionary containing average losses for the epoch:
            - 'loss': Total loss (VAE + classification) averaged over batches.
            - 'recon': Reconstruction loss averaged over batches.
            - 'kl': KL divergence loss averaged over batches.
            - 'class_loss': Classification loss averaged over batches.

    Raises:
        RuntimeError: If model forward or optimizer step fails.
        KeyError: If required keys ('alpha', 'gamma') are missing in config.
    """
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss, total_recon, total_kl, total_class = 0, 0, 0, 0

    # Context manager handles torch.no_grad() for evaluation
    with torch.set_grad_enabled(is_training):
        for condition, expression, moa_label in loader:
            expression, condition, moa_label = (
                expression.to(device),
                condition.to(device),
                moa_label.to(device),
            )

            if is_training:
                optimizer.zero_grad()

            recon_x, mu, log_var = model(expression, condition)
            moa_prediction = model.classify(mu)

            loss, recon, kl = vae_loss_function(
                recon_x,
                expression,
                mu,
                log_var,
                model.latent_dim,
                beta,
                recon_weight=config["alpha"],
            )
            mask = moa_label != -1
            class_loss = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                class_loss = nn.functional.cross_entropy(
                    moa_prediction[mask], moa_label[mask]
                )

            total_epoch_loss = loss + config["gamma"] * class_loss

            if is_training:
                total_epoch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += total_epoch_loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_class += class_loss.item()

    # Return average losses
    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "kl": total_kl / num_batches,
        "class_loss": total_class / num_batches,
    }


def create_animation_frames(
    epoch_history: list[dict],
    animation_dir: str,
) -> list[str]:
    """
    Creates PCA plots for each epoch and saves them as image frames for animation.

    Args:
        epoch_history (list[dict]): List of dictionaries containing embeddings and labels for each epoch.
        animation_dir (str): Directory to save the animation frames.
        df_meta_test (pd.DataFrame): Metadata DataFrame for the test set, used for coloring points.

    Returns:
        list[str]: List of file paths to the saved frame images.
    """
    # Fit PCA on the final epoch's embeddings for a stable projection
    final_embeddings = epoch_history[-1]["embeddings"]
    pca = PCA(n_components=2, random_state=42)
    pca.fit(final_embeddings)

    all_embeddings: list[np.ndarray] = []
    for epoch_data in epoch_history:
        # Use the SAME fitted PCA to transform embeddings from every epoch
        projected_embeddings = pca.transform(epoch_data["embeddings"])
        all_embeddings.append(projected_embeddings)

    stacked_embeddings = np.concatenate(all_embeddings, axis=0)
    x_lim = (stacked_embeddings[:, 0].min() - 1, stacked_embeddings[:, 0].max() + 1)
    y_lim = (stacked_embeddings[:, 1].min() - 1, stacked_embeddings[:, 1].max() + 1)

    labels_for_legend = epoch_history[0]["labels"]

    le = LabelEncoder()
    label_ids = le.fit_transform(labels_for_legend)

    frame_files: list[str] = []
    for i, epoch_data in enumerate(epoch_history):
        epoch_num = i + 1

        fig, axes = plt.subplots(figsize=(10, 10))
        scatter = axes.scatter(
            all_embeddings[i][:, 0],
            all_embeddings[i][:, 1],
            c=label_ids,
            alpha=0.5,
            cmap="tab10",
            s=40,
        )
        handles, _ = scatter.legend_elements()
        axes.legend(
            handles=handles,
            labels=list(le.classes_),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        axes.grid(False)
        axes.set_title(f"Latent Space PCA - Epoch {epoch_num}")
        axes.set_xlabel("PCA Component 1")
        axes.set_ylabel("PCA Component 2")
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)

        axes.set_aspect("equal", adjustable="box")

        # Save the frame
        frame_path = os.path.join(animation_dir, f"epoch_{epoch_num:03d}.png")
        plt.savefig(frame_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        frame_files.append(frame_path)

    return frame_files


def build_animation_gif(
    frame_files: list[str],
    output_path: str,
    duration: float = 0.5,
) -> None:
    """
    Builds a GIF animation from a list of image frame files.

    Args:
        frame_files (list[str]): List of file paths to image frames.
        output_path (str): Path to save the output GIF file.
        duration (float, optional): Duration (seconds) per frame in the GIF. Defaults to 0.5.
    """
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)


def generate_embeddings(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate latent embeddings (mu) and labels for a given dataset using a DataLoader.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for condition, expression, moa_label in loader:
            # Note: We only need expression and condition for the model input
            expression, condition = expression.to(device), condition.to(device)

            # Forward pass to get the mean of the latent space
            _, mu, _ = model(expression, condition)

            all_embeddings.append(mu.cpu().numpy())
            all_labels.append(moa_label.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels


def get_data(processed_data_path, batch_size) -> Tuple:
    """
    Loads processed training, validation, and test data, along with metadata and mappings.

    Returns:
        tuple: (
            train_loader (DataLoader): DataLoader for training set,
            val_loader (DataLoader): DataLoader for validation set,
            test_loader (DataLoader): DataLoader for test set,
            df_train (pd.DataFrame): Training gene expression data,
            df_val (pd.DataFrame): Validation gene expression data,
            df_test (pd.DataFrame): Test gene expression data,
            df_meta_train (pd.DataFrame): Training metadata,
            df_meta_val (pd.DataFrame): Validation metadata,
            df_meta_test (pd.DataFrame): Test metadata,
            df_gene_mapping (pd.DataFrame): Gene mapping DataFrame,
            int_to_moa (Dict[int, str]): Mapping from integer MOA labels to strings
        )
    """
    df_train = pd.read_parquet(os.path.join(processed_data_path, "train_data.parquet"))
    df_val = pd.read_parquet(os.path.join(processed_data_path, "val_data.parquet"))
    df_test = pd.read_parquet(os.path.join(processed_data_path, "test_data.parquet"))

    df_meta_train = pd.read_parquet(
        os.path.join(processed_data_path, "train_meta.parquet")
    )
    df_meta_val = pd.read_parquet(os.path.join(processed_data_path, "val_meta.parquet"))
    df_meta_test = pd.read_parquet(
        os.path.join(processed_data_path, "test_meta.parquet")
    )

    # load the gene mapping
    df_gene_mapping = pd.read_parquet(
        os.path.join(processed_data_path, "gene_mapping.parquet")
    )

    # Load the int_to_moa mapping
    with open(os.path.join(processed_data_path, "int_to_moa.yaml"), "r") as f:
        int_to_moa = yaml.safe_load(f)

    train_dataset = LincsDataset(df_train, df_meta_train)
    val_dataset = LincsDataset(df_val, df_meta_val)
    test_dataset = LincsDataset(df_test, df_meta_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        df_train,
        df_val,
        df_test,
        df_meta_train,
        df_meta_val,
        df_meta_test,
        df_gene_mapping,
        int_to_moa,
    )


def get_model_weights(
    expression_dim: int,
    num_classes: int,
    conditional_dim: int,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> CVAE:
    """
    Initializes and returns a CVAE model with the specified configuration.

    Args:
        expression_dim (int): Number of gene expression features.
        num_classes (int): Number of MOA classes.
        conditional_dim (int): Dimension of the condition (fingerprint) vector.
        config (Dict[str, Any]): Dictionary containing model hyperparameters.

    Returns:
        CVAE: Initialized CVAE model on the appropriate device.
    """
    model = CVAE(
        expression_dim=expression_dim,
        num_classes=num_classes,
        condition_dim=conditional_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_encoder_layers=config["num_encoder_layers"],
        encoder_dropout_rate=config["encoder_dropout"],
        num_decoder_layers=config["num_decoder_layers"],
        condition_emb_dim=config["condition_emb_dim"],
        decoder_dropout_rate=config["decoder_dropout"],
        num_molecular_emb_layers=config["num_molecular_emb_layers"],
    )

    model = model.to(device)

    return model


def setup_directories(
    config: Dict[str, Any], run: wandb.sdk.wandb_run.Run
) -> Tuple[str, str]:
    """
    Sets up directories for saving images and animation frames during training.

    Args:
        config (Dict[str, Any]): Dictionary containing model and training hyperparameters.
        run (wandb.sdk.wandb_run.Run): The current Weights & Biases run object.

    Returns:
        Tuple[str, str]:
            - img_save_path: Path to save images and results for the run.
            - animation_save_path: Path to save animation frame images.
    """
    # Use the immutable run ID for the directory name
    run_id = run.id
    run_name = run.name.split("-")[-1] if run.name else "default_run"
    # Use the sweep ID to group sweep runs, or use a default for single runs
    base_path_name = run.sweep_id if run.sweep_id else "single_runs"

    # Create a descriptive directory name using the unique ID
    img_save_dir = f"{run_name}_{run.name}_{run_id}_{config['latent_dim']}latent_{config['hidden_dim']}hidden_{config['num_encoder_layers']}enclayers_{config['encoder_dropout']:.2f}do_{config['condition_emb_dim']}condembs_{config['learning_rate']:.2f}lr_{config['alpha']:.2f}a_{config['beta']:.2f}b_{config['gamma']:.2f}g_{config['num_molecular_emb_layers']}condlayers_{config['num_decoder_layers']}declayers"

    img_save_path = os.path.join(
        "data", "hyperparameter_runs", base_path_name, img_save_dir
    )
    animation_save_path = os.path.join(img_save_path, "animation_frames")
    os.makedirs(animation_save_path, exist_ok=True)

    return img_save_path, animation_save_path


def training_loop(
    config: Dict[str, Any],
    model: CVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    int_to_moa: Dict[int, str],
    df_meta_test: pd.DataFrame,
    wandb: wandb.sdk.wandb_run.Run,
    device: torch.device,
) -> Tuple[CVAE, Dict[str, List[float]], List[dict]]:
    """
    Main training loop for the CVAE model.
    Args:
        config (Dict[str, Any]): Configuration dictionary containing model and training hyperparameters.
        model (CVAE): The CVAE model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run computations on (e.g., 'cpu' or 'cuda').
    """

    if torch.cuda.is_available():
        # Initialize BnB
        optimizer = bnb.Adam8bit(
            model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
        )

    history = {
        k: []
        for k in [
            "train_loss",
            "val_loss",
            "train_recon",
            "val_recon",
            "train_kl",
            "val_kl",
            "train_class_loss",
            "val_class_loss",
        ]
    }
    animation_data = []

    # Main Training Loop
    pbar_epochs = tqdm(range(1, config["num_epochs"] + 1), desc="Overall Progress")
    for epoch in pbar_epochs:
        beta = (
            config["beta"] * ((epoch - 1) / config["num_epochs"])
            if config.get("beta_anneal")
            else config["beta"]
        )

        train_metrics = _run_epoch(
            model, train_loader, config, beta, device, optimizer=optimizer
        )
        val_metrics = _run_epoch(model, val_loader, config, beta, device)

        for key in train_metrics:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        wandb.log(
            {
                "Train Loss": train_metrics["loss"],
                "Val Loss": val_metrics["loss"],
                "Train Recon": train_metrics["recon"],
                "Val Recon": val_metrics["recon"],
                "Train KL": train_metrics["kl"],
                "Val KL": val_metrics["kl"],
                "Train Class Loss": train_metrics["class_loss"],
                "Val Class Loss": val_metrics["class_loss"],
                "Beta": beta,
                "Epoch": epoch,
            },
            step=epoch,
        )

        # Generate and store data for animation
        test_embeddings, all_moa_labels = generate_embeddings(
            model=model,
            loader=test_loader,
            device=device,
        )
        moa_labels = [int_to_moa[int_label] for int_label in all_moa_labels.tolist()]
        animation_data.append(
            {
                "embeddings": test_embeddings,
                "labels": moa_labels,
                "pert_labels": df_meta_test["pert_id"].tolist(),
            }
        )
        pbar_epochs.set_postfix(
            {
                "Train Loss": f"{train_metrics['loss']:.4f}",
                "Val Loss": f"{val_metrics['loss']:.4f}",
            }
        )

    return model, history, animation_data

def run_training_pipeline(
    config: Dict[str, Any],
    processed_data_path: str,
    batch_size: int,
    device: torch.device,
    is_sweep: bool = False,
) -> None:
    """
    Runs the training pipeline for the CVAE model, including training, validation, logging,
    animation creation, and evaluation.

    Args:
        config (Dict[str, Any]): Dictionary containing model and training hyperparameters.
        is_sweep (bool, optional): Whether the run is part of a hyperparameter sweep. Defaults to False.

    """
    wandb.init(project="drug-induced-gene-expression-prediction", config=config)
    config = wandb.config
    (
        train_loader,
        val_loader,
        test_loader,
        df_train,
        _,
        _,
        df_meta_train,
        _,
        df_meta_test,
        df_gene_mapping,
        int_to_moa,
    ) = get_data(processed_data_path, batch_size)

    # Initialize the model with the specified configuration
    model = get_model_weights(
        expression_dim=df_train.shape[1],
        num_classes=df_meta_train["moa_int"].nunique(),
        conditional_dim=df_meta_train["fingerprint"].iloc[0].shape[0],
        config=config,
        device=device,
    )

    # Set up directories for saving images and animation frames
    img_save_path, animation_save_path = setup_directories(config, wandb.run)

    # Run the training loop
    model, history, animation_data = training_loop(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        df_meta_test=df_meta_test,
        int_to_moa=int_to_moa,
        wandb=wandb,
        device=device,
    )
    
    gsea_gen_error_list, gsea_recon_error_list = [], []
    for (pert_id, drug_name) in [
        ("BRD-A00993607", "Bortezomib"),
        ("BRD-K01800709", "Trichostatin A"),
        ("BRD-K29699988", "Erlotinib")
        ]:
        gsea_recon_error_item, gsea_gen_error_item = evaluate_recon_and_gen_gsea_for_pert(
            pert_id_to_test=pert_id,
            drug_name=drug_name,
            val_dataset=val_loader.dataset,
            df_gene_mapping=df_gene_mapping,
            img_save_path=img_save_path,
            model=model,
            device=device,
        )
        gsea_recon_error_list.append(gsea_recon_error_item)
        gsea_gen_error_list.append(gsea_gen_error_item)

    gsea_recon_error = np.mean(gsea_recon_error_list)
    gsea_gen_error = np.mean(gsea_gen_error_list)

    train_corr, val_corr = get_recon_correlation(
        model, train_loader, val_loader, img_save_path, device=device
    )

    total_error_to_minimize = (
        gsea_recon_error + gsea_gen_error + (1 - train_corr) + (1 - val_corr)
    )
    wandb.log(
        {
            "gsea_gen_error": gsea_gen_error,
            "gsea_recon_error": gsea_recon_error,
            "train_corr": train_corr,
            "val_corr": val_corr,
            "total_error_to_minimize": total_error_to_minimize,
        }
    )
    # Create and Log Animation
    frame_files = create_animation_frames(animation_data, animation_save_path)
    animation_path = os.path.join(animation_save_path, "training_animation.gif")
    build_animation_gif(frame_files, animation_path, duration=0.5)
    wandb.log({"training_animation": wandb.Video(animation_path, fps=2, format="gif")})
    plot_training_history(history)
    plt.savefig(os.path.join(img_save_path, "training_history.png"))

    if is_sweep:
        wandb.finish()

    return model
