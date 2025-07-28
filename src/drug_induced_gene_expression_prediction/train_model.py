import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
from tqdm import tqdm

import wandb
import imageio

import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import seaborn as sns

from drug_induced_gene_expression_prediction import (
    LincsDataset,
    CVAE,
    vae_loss_function,
    evaluate_recon_and_gen_gsea_for_pert,
    plot_training_history,
    get_recon_correlation,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCESSED_DATA_PATH = os.path.join(
    "data", "processed", "drug_induced_gene_expression_prediction"
)

BATCH_SIZE = 64


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss, total_recon, total_kl, total_class = 0, 0, 0, 0

    for condition, expression, moa_label in loader:
        expression, condition, moa_label = (
            expression.to(device),
            condition.to(device),
            moa_label.to(device),
        )

        optimizer.zero_grad()
        recon_x, mu, log_var = model(expression, condition)
        moa_prediction = model.classify(mu)

        # Calculate VAE and classification loss
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
        total_epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_epoch_loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_class += class_loss.item()

    # Return average losses for the epoch
    return {
        "loss": total_loss / len(loader),
        "recon": total_recon / len(loader),
        "kl": total_kl / len(loader),
        "class_loss": total_class / len(loader),
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: Dict[str, Any],
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a single validation/evaluation epoch."""
    model.eval()
    total_loss, total_recon, total_kl, total_class = 0, 0, 0, 0

    with torch.no_grad():
        for condition, expression, moa_label in loader:
            expression, condition, moa_label = (
                expression.to(device),
                condition.to(device),
                moa_label.to(device),
            )
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

            total_loss += total_epoch_loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_class += class_loss.item()

    return {
        "loss": total_loss / len(loader),
        "recon": total_recon / len(loader),
        "kl": total_kl / len(loader),
        "class_loss": total_class / len(loader),
    }


def create_animation_frames(
    epoch_history: list,
    animation_dir: str,
    df_meta_test: pd.DataFrame,
) -> list:
    """Creates PCA plots for each epoch and saves them as frames."""
    print("Generating animation frames...")
    # Fit PCA on the final epoch's embeddings for a stable projection
    final_embeddings = epoch_history[-1]["embeddings"]
    pca = PCA(n_components=2, random_state=42)
    pca.fit(final_embeddings)

    frame_files = []
    all_embeddings = []
    for i, epoch_data in enumerate(epoch_history):
        # Use the SAME fitted PCA to transform embeddings from every epoch
        projected_embeddings = pca.transform(epoch_data["embeddings"])
        all_embeddings.append(projected_embeddings)

    stacked_embeddings = np.concatenate(all_embeddings, axis=0)
    x_lim = (stacked_embeddings[:, 0].min() - 1, stacked_embeddings[:, 0].max() + 1)
    y_lim = (stacked_embeddings[:, 1].min() - 1, stacked_embeddings[:, 1].max() + 1)

    labels_for_legend = epoch_history[0]["pert_labels"]
    unique_labels = np.unique(labels_for_legend)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]

    legend_elements_pert = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Class {label}",
            markerfacecolor=color,
            markersize=8,
            linewidth=0,
            alpha=0.7,
        )
        for label, color in zip(unique_labels, colors)
    ]

    labels_for_legend = epoch_history[0]["labels"]
    unique_labels = np.unique(labels_for_legend)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]

    legend_elements_ids = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Class {label}",
            markerfacecolor=color,
            markersize=8,
            linewidth=0,
            alpha=0.7,
        )
        for label, color in zip(unique_labels, colors)
    ]

    for i, epoch_data in enumerate(epoch_history):
        epoch_num = i + 1

        fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)

        sns.scatterplot(
            x=all_embeddings[i][:, 0],
            y=all_embeddings[i][:, 1],
            hue=df_meta_test["pert_id"].tolist(),
            palette="tab20",
            s=40,
            alpha=0.5,
            ax=ax[0],
        )
        ax[0].set_title(f"Latent Space PCA - Epoch {epoch_num}")
        ax[0].set_xlabel("PCA Component 1")
        ax[0].set_ylabel("PCA Component 2")
        ax[0].set_xlim(x_lim)
        ax[0].set_ylim(y_lim)

        # Add legend
        ax[0].legend(
            handles=legend_elements_pert,
            title="Classes",
            fontsize=12,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        ax[0].set_aspect("equal", adjustable="box")

        sns.scatterplot(
            x=all_embeddings[i][:, 0],
            y=all_embeddings[i][:, 1],
            hue=df_meta_test["pert_id"].tolist(),
            palette="tab10",
            s=40,
            alpha=0.5,
            ax=ax[1],
        )
        ax[1].set_title(f"Latent Space PCA - Epoch {epoch_num}")
        ax[1].set_xlabel("PCA Component 1")
        ax[1].set_ylabel("PCA Component 2")
        ax[1].set_xlim(x_lim)

        # Add legend
        ax[1].legend(
            handles=legend_elements_ids,
            title="Classes",
            fontsize=12,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        ax[1].set_aspect("equal", adjustable="box")

        # Save the frame
        frame_path = os.path.join(animation_dir, f"epoch_{epoch_num:03d}.png")
        plt.savefig(frame_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        frame_files.append(frame_path)

    return frame_files


def build_animation_gif(
    frame_files: list,
    output_path: str,
    duration: float = 0.5,
):
    """Builds a GIF from a list of frame files."""
    print("Creating animation...")
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Animation saved to {output_path}")


def plot_pca(
    embeddings: np.ndarray, labels: pd.Series, title: str = "PCA Plot", axes=None
) -> None:
    """
    Plot PCA embeddings with color-coded labels.

    Args:
        embeddings (np.ndarray): embeddings.
        labels (pd.Series): Labels for coloring the points.
        title (str): Title of the plot.
        axes (plt.Axes, optional): Axes to plot on. If None, creates a new figure.
    """
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    if axes is None:
        _, axes = plt.subplots(figsize=(8, 6))

    # Encode labels for coloring
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    pca_model = PCA(n_components=2, random_state=42)
    embedding = pca_model.fit_transform(embeddings)
    # Plot
    scatter = axes.scatter(
        embedding[:, 0], embedding[:, 1], c=label_ids, cmap="tab10", s=40
    )
    axes.set_title(title)
    # Fix legend for UMAP plot with multiple classes
    handles, _ = scatter.legend_elements()
    axes.legend(
        handles=handles,
        labels=list(le.classes_),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()

    return axes


def generate_embeddings(model, dataset, df_meta_data, device, int_to_moa, batch_size):
    """
    Generate latent embeddings (mu) and labels for a given dataset.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        # for condition, expression, moa_label in loader:
        meta_batches = [
            df_meta_data.iloc[i : i + batch_size]
            for i in range(0, len(df_meta_data), batch_size)
        ]
        expression_batches = [
            dataset.df_expression.iloc[i : i + batch_size]
            for i in range(0, len(dataset.df_expression), batch_size)
        ]

        for meta_batch, expression_batch in zip(meta_batches, expression_batches):
            conditions = torch.tensor(np.stack(meta_batch["fingerprint"].values)).to(
                device
            )
            # moa_labels = torch.tensor(meta_batch["pert_id"].values).to(device)
            moa_labels = torch.tensor(meta_batch["moa_int"].values).to(device)
            expressions = torch.tensor(expression_batch.values).to(device)

            # Forward pass
            _, mu, _ = model(expressions, conditions)

            all_embeddings.append(mu.cpu().numpy())
            all_labels.append(moa_labels.cpu().numpy())

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_moa_labels = [int_to_moa[label] for label in all_labels]
    return (
        all_embeddings,
        all_labels,
        all_moa_labels,
    )


def get_data():
    df_train = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "train_data.parquet"))
    df_val = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "val_data.parquet"))
    df_test = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "test_data.parquet"))

    df_meta_train = pd.read_parquet(
        os.path.join(PROCESSED_DATA_PATH, "train_meta.parquet")
    )
    df_meta_val = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, "val_meta.parquet"))
    df_meta_test = pd.read_parquet(
        os.path.join(PROCESSED_DATA_PATH, "test_meta.parquet")
    )

    # load the gene mapping
    df_gene_mapping = pd.read_parquet(
        os.path.join(PROCESSED_DATA_PATH, "gene_mapping.parquet")
    )

    # Load the int_to_moa mapping
    with open(os.path.join(PROCESSED_DATA_PATH, "int_to_moa.yaml"), "r") as f:
        int_to_moa = yaml.safe_load(f)

    train_dataset = LincsDataset(df_train, df_meta_train)
    val_dataset = LincsDataset(df_val, df_meta_val)
    test_dataset = LincsDataset(df_test, df_meta_test)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


def get_model_weights():
    (
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
    ) = get_data()

    # read in training config
    with open(
        os.path.join("config", "drug_induced_gene_expression_prediction.yaml"),
        "r",
    ) as file:
        config = yaml.safe_load(file)

    # # Initialize the CVAE model
    model = CVAE(
        expression_dim=df_train.shape[1],  # Number of genes
        num_classes=df_meta_train[
            "moa_int"
        ].nunique(),  # Number of MOA classes + 1 for "Other"
        condition_dim=df_meta_train["fingerprint"].iloc[0].shape[0],  # Fingerprint size
        hidden_dim=config["hidden_dim"],  # Hidden dimension for the encoder and decoder
        latent_dim=config["latent_dim"],  # Latent space dimension
        num_encoder_layers=config[
            "num_encoder_layers"
        ],  # Number of layers in the encoder
        encoder_dropout_rate=config["encoder_dropout"],  # Dropout rate for the encoder
        num_decoder_layers=config[
            "num_decoder_layers"
        ],  # Number of layers in the decoder
        condition_emb_dim=config[
            "condition_emb_dim"
        ],  # Embedding dimension for the condition (fingerprint)
        decoder_dropout_rate=config["decoder_dropout"],  # Dropout rate for the decoder
        num_molecular_emb_layers=config[
            "num_molecular_emb_layers"
        ],  # Number of layers in the molecular embedding
    )

    model = model.to(DEVICE)

    return (
        model,
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
        int_to_moa
    )


def train_model():
    # read in training config
    with open(os.path.join("config", "sweep.yaml"), "r") as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(
        sweep_config, project="drug_induced_gene_expression_prediction"
    )
    (
        model,
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
        int_to_moa
    ) = get_model_weights()

    def train_with_sweep():
        with wandb.init() as run:
            config = wandb.config

            model = CVAE(
                expression_dim=df_train.shape[1],  # Number of genes
                num_classes=df_meta_train[
                    "moa_int"
                ].nunique(),  # Number of MOA classes + 1 for "Other"
                condition_dim=df_meta_train["fingerprint"]
                .iloc[0]
                .shape[0],  # Fingerprint size
                hidden_dim=config[
                    "hidden_dim"
                ],  # Hidden dimension for the encoder and decoder
                latent_dim=config["latent_dim"],  # Latent space dimension
                num_encoder_layers=config[
                    "num_encoder_layers"
                ],  # Number of layers in the encoder
                encoder_dropout_rate=config[
                    "encoder_dropout"
                ],  # Dropout rate for the encoder
                num_decoder_layers=config[
                    "num_decoder_layers"
                ],  # Number of layers in the decoder
                condition_emb_dim=config[
                    "condition_emb_dim"
                ],  # Embedding dimension for the condition (fingerprint)
                decoder_dropout_rate=config[
                    "decoder_dropout"
                ],  # Dropout rate for the decoder
                num_molecular_emb_layers=config[
                    "num_molecular_emb_layers"
                ],  # Number of layers in the molecular embedding
            )
            model = model.to(DEVICE)

            num_epochs = config["num_epochs"]
            alpha = config["alpha"]  # Reconstruction loss weight
            gamma = config["gamma"]  # Classification loss weight
            beta = config["beta"]  # KL divergence weight

            sweep_name = run.sweep_id
            run_name = run.name
            run_id = run.name.split("-")[-1]
            img_save_dir = f"{run_id}_{run_name}_cvae_model_{config['latent_dim']}latent_{config['hidden_dim']}hidden_{config['num_encoder_layers']}enclayers_{config['encoder_dropout']:.2f}do_{config['condition_emb_dim']}condembs_{config['learning_rate']:.2f}lr_{config['alpha']:.2f}a_{config['beta']:.2f}b_{config['gamma']:.2f}g_{config['num_molecular_emb_layers']}condlayers_{config['num_decoder_layers']}declayers"
            img_save_path = os.path.join(
                "data", "hyperparameter_runs", sweep_name, img_save_dir
            )
            animation_save_path = os.path.join(
                "data", "hyperparameter_runs", sweep_name, img_save_dir, "animation_frames"
            )

            os.makedirs(animation_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)
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
            pbar_epochs = tqdm(
                range(1, config["num_epochs"] + 1), desc="Overall Progress"
            )
            for epoch in pbar_epochs:
                beta = (
                    config["beta"] * ((epoch - 1) / config["num_epochs"])
                    if config.get("beta_anneal")
                    else config["beta"]
                )

                train_metrics = train_epoch(
                    model, train_loader, optimizer, config, beta, DEVICE
                )
                val_metrics = evaluate_epoch(model, val_loader, config, beta, DEVICE)

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
                test_embeddings, test_labels, _ = generate_embeddings(
                    model=model,
                    dataset=test_loader.dataset,
                    df_meta_data=df_meta_test,
                    device=DEVICE,
                    int_to_moa=int_to_moa,
                    batch_size=test_loader.batch_size,
                )
                animation_data.append(
                    {
                        "embeddings": test_embeddings,
                        "labels": test_labels,
                        "pert_labels": df_meta_test["pert_id"].tolist(),
                    }
                )
                pbar_epochs.set_postfix(
                    {
                        "Train Loss": f"{train_metrics['loss']:.4f}",
                        "Val Loss": f"{val_metrics['loss']:.4f}",
                    }
                )

            gsea_recon_error, gsea_gen_error = evaluate_recon_and_gen_gsea_for_pert(
                pert_id_to_test="BRD-A00993607",
                val_dataset=val_loader.dataset,
                df_gene_mapping=df_gene_mapping,
                img_save_path=img_save_path,
                model=model,
                device=DEVICE,
            )

            train_corr, val_corr = get_recon_correlation(
                model, train_loader, val_loader, img_save_path, device=DEVICE
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
            frame_files = create_animation_frames(
                animation_data, animation_save_path, df_meta_test
            )
            animation_path = os.path.join(
                animation_save_path, "training_animation.gif"
            )
            build_animation_gif(frame_files, animation_path, duration=0.5)
            wandb.log(
                {"training_animation": wandb.Video(animation_path, fps=2, format="gif")}
            )
            plot_training_history(history)
            plt.savefig(os.path.join(img_save_path, "training_history.png"))

            # Cleanup and Return
            wandb.finish()
            return model, history

    wandb.agent(sweep_id, function=train_with_sweep, count=100)


if __name__ == "__main__":
    # Initialize the model and start training
    train_model()
