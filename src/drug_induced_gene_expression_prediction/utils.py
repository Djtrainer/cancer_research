import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
from tqdm import tqdm

import wandb

from .eval import evaluate_recon_and_gen_gsea_for_pert, get_recon_correlation

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


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
    epoch: int,
) -> Tuple[nn.Module, float, float, float, float]:
    """
    Perform one epoch of training for the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Total number of epochs.
        alpha (float): Reconstruction loss weight.
        beta (float): KL divergence loss weight.
        gamma (float): Classification loss weight.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        Tuple[nn.Module, float, float, float, float]:
            Updated model, total train loss, reconstruction loss, KL loss, classification loss.
    """
    train_loss, train_recon, train_kl, train_class_loss = 0, 0, 0, 0

    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for condition, expression, moa_label in pbar_train:
        expression = expression.to(device)
        condition = condition.to(device)
        moa_label = moa_label.to(device)

        model.train()
        optimizer.zero_grad()

        # Forward pass
        recon_x, mu, log_var = model(expression, condition)
        moa_prediction = model.classify(mu)
        # mask for only those MOAs that are known
        mask = moa_label != -1
        epoch_class_loss = torch.tensor(0.0, device=device)
        if mask.sum() > 0:
            labeled_preds = moa_prediction[mask]
            labeled_labels = moa_label[mask]
            epoch_class_loss = nn.functional.cross_entropy(
                labeled_preds, labeled_labels
            )

        loss, recon, kl = vae_loss_function(
            recon_x, expression, mu, log_var, model.latent_dim, beta, recon_weight=alpha
        )

        # Add classification loss to the total loss
        loss += gamma * epoch_class_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()
        train_class_loss += epoch_class_loss.item()

        pbar_train.set_description(
            f"Epoch {epoch:02d} | "
            f"Total Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Train Reconstruction Loss: {train_recon / len(train_loader):.4f} | "
            f"Train Classification Loss: {train_class_loss / len(train_loader):.4f} | "
            f"Train KL Divergence: {train_kl / len(train_loader):.4f} | "
            f"Beta: {beta:.4f} | "
            f"Gamma: {gamma:.4f}"
        )

    return model, train_loss, train_recon, train_kl, train_class_loss


def validation_loop(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    alpha: float,
    beta: float,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float, float, float]:
    """
    Perform one epoch of validation for the model.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run validation on.
        alpha (float): Reconstruction loss weight.
        beta (float): KL divergence loss weight.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        Tuple[float, float, float, float]:
            Total validation loss, reconstruction loss, KL loss, classification loss.
    """
    model.eval()
    val_loss, val_recon, val_kl, val_class_loss = 0, 0, 0, 0

    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} Validation")
    with torch.no_grad():
        for condition, expression, moa_label in pbar_val:
            expression = expression.to(device)
            condition = condition.to(device)
            moa_label = moa_label.to(device)

            recon_x, mu, log_var = model(expression, condition)
            moa_prediction = model.classify(mu)

            loss, recon, kl = vae_loss_function(
                recon_x,
                expression,
                mu,
                log_var,
                model.latent_dim,
                beta,
                recon_weight=alpha,
            )

            mask = moa_label != -1
            epoch_class_loss = 0
            if mask.sum() > 0:
                labeled_preds = moa_prediction[mask]
                labeled_labels = moa_label[mask]
                epoch_class_loss = nn.functional.cross_entropy(
                    labeled_preds, labeled_labels
                )

            val_loss += loss.item()
            val_recon += recon.item()
            val_kl += kl.item()
            val_class_loss += epoch_class_loss.item()

            pbar_val.set_description(
                f"Epoch {epoch:02d} | "
                f"Total Val Loss: {val_loss / len(val_loader):.4f} | "
                f"Val Reconstruction Loss: {val_recon / len(val_loader):.4f} | "
                f"Val Classification Loss: {val_class_loss / len(val_loader):.4f} | "
                f"Val KL Divergence: {val_kl / len(val_loader):.4f} | "
            )

    return val_loss, val_recon, val_kl, val_class_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    df_gene: pd.DataFrame = None,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train the Conditional Variational Autoencoder (CVAE) model.

    Args:
        model (nn.Module): The CVAE model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        config (Dict[str, Any]): Configuration dictionary containing training parameters.
        device (torch.device, optional): Device to run the training on (CPU or GPU).
        beta_anneal (bool, optional): Whether to anneal beta over epochs. Defaults to False.
        df_gene (pd.DataFrame, optional): DataFrame containing gene information for evaluation.

    Returns:
        Tuple[nn.Module, Dict[str, list]]:
            The trained model and a dictionary containing training history.
    """
    num_epochs = config["num_epochs"]
    alpha = config["alpha"]  # Reconstruction loss weight
    gamma = config["gamma"]  # Classification loss weight
    beta = config["beta"]  # KL divergence weight

    # model_save_dir = os.path.join(ROOT, config["model_save_dir"])
    # model_save_name = f"cvae_model_{config['latent_dim']}latent_{config['hidden_dim']}hidden_{config['num_encoder_layers']}enclayers_{config['encoder_dropout']}do_{config['condition_emb_dim']}condembs_{config['learning_rate']}lr_{config['alpha']}a_{config['beta']}b_{config['num_molecular_emb_layers']}condlayers_{config['num_decoder_layers']}declayers.pth"
    # model_save_path = os.path.join(model_save_dir, model_save_name)
    # os.makedirs(model_save_dir, exist_ok=True)
    img_save_dir = f"cvae_model_{config['latent_dim']}latent_{config['hidden_dim']}hidden_{config['num_encoder_layers']}enclayers_{config['encoder_dropout']:.2f}do_{config['condition_emb_dim']}condembs_{config['learning_rate']:.2f}lr_{config['alpha']}a_{config['beta']}b_{config['num_molecular_emb_layers']}condlayers_{config['num_decoder_layers']}declayers"
    img_save_path = os.path.join(ROOT, "imgs", img_save_dir)

    os.makedirs(img_save_path, exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    model = model.to(device)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
        "train_class_loss": [],
        "val_class_loss": [],
    }

    best_val_loss = float("inf")  # Initialize best validation loss to infinity
    beta_ = beta  # Initialize beta for annealing if applicable
    for epoch in range(1, num_epochs + 1):
        if config["beta_anneal"]:
            beta_ = beta * ((epoch - 1) / num_epochs)
        # Training loop
        model, train_loss, train_recon, train_kl, train_class_loss = training_loop(
            model,
            train_loader,
            optimizer,
            num_epochs,
            alpha,
            beta_,
            gamma,
            device,
            epoch,
        )
        # Validation loop
        val_loss, val_recon, val_kl, val_class_loss = validation_loop(
            model, val_loader, device, alpha, beta_, epoch, num_epochs
        )

        # Save the model if the validation loss is the best so far
        # if val_loss / len(val_loader) < best_val_loss:
        #     best_val_loss = val_loss / len(val_loader)
        #     torch.save(model.state_dict(), model_save_path)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_recon"].append(train_recon / len(train_loader))
        history["val_recon"].append(val_recon / len(val_loader))
        history["train_kl"].append(train_kl / len(train_loader))
        history["val_kl"].append(val_kl / len(val_loader))
        history["train_class_loss"].append(train_class_loss / len(train_loader))
        history["val_class_loss"].append(val_class_loss / len(val_loader))

        wandb.log(
            {
                "Total Train Loss": avg_train_loss,
                "Total Val Loss": avg_val_loss,
                "Recon Train Loss": train_recon / len(train_loader),
                "Recon Val Loss": val_recon / len(val_loader),
                "KL Train": train_kl / len(train_loader),
                "KL Val": val_kl / len(val_loader),
                "Train Classification Loss": train_class_loss / len(train_loader),
                "Val Classification Loss": val_class_loss / len(val_loader),
                "Beta": beta_,
                "Gamma": gamma,
            },
            step=epoch,
        )

    gsea_recon_error, gsea_gen_error = evaluate_recon_and_gen_gsea_for_pert(
        pert_id_to_test="BRD-A00993607",
        pert_id_control="DMSO",
        val_dataset=val_loader.dataset,
        df_gene=df_gene,
        img_save_path=img_save_path,
        model=model,
    )

    train_corr, val_corr = get_recon_correlation(
        model, train_loader, val_loader, img_save_path
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

    wandb.finish()

    return model, history


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