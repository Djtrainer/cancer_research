from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Encoder(torch.nn.Module):
    """Encoder for the Conditional Variational Autoencoder (CVAE).

    Encodes input data into latent space parameters (mean and log-variance).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 0,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
            num_layers (int, optional): Number of hidden layers. Defaults to 0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Encoder, self).__init__()

        self.fc_initial = nn.Linear(input_dim, hidden_dim * 4 * (num_layers + 1) ** 2)

        self.fc_layers = nn.ModuleList()
        if num_layers > 0:
            for layer_mult in range(1, num_layers + 1)[::-1]:
                self.fc_layers.append(
                    nn.Linear(
                        hidden_dim * (layer_mult + 1) ** 2 * 4,
                        hidden_dim * layer_mult**2 * 4,
                    )
                )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log-variance tensors.
        """
        x = torch.relu(self.fc_initial(x))
        x = self.dropout(x)

        for fc_layer in self.fc_layers:
            x = torch.relu(fc_layer(x))
            x = self.dropout(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Clamp logvar to a reasonable range to prevent numerical overflow/underflow.
        logvar = torch.clamp(logvar, min=-20, max=20)

        return mu, logvar


class Decoder(torch.nn.Module):
    """Decoder for the Conditional Variational Autoencoder (CVAE).

    Decodes latent variables and condition embeddings to reconstruct input data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 0,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Args:
            input_dim (int): Dimension of the input latent vector.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
            num_layers (int, optional): Number of hidden layers. Defaults to 0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc_initial = nn.Linear(input_dim, hidden_dim * 4)

        self.fc_layers = nn.ModuleList()
        if num_layers > 0:
            for layer_mult in range(1, num_layers + 1):
                self.fc_layers.append(
                    nn.Sequential(
                        nn.Linear(
                            hidden_dim * layer_mult**2 * 4,
                            hidden_dim * (layer_mult + 1) ** 2 * 4,
                        ),
                        nn.BatchNorm1d(hidden_dim * (layer_mult + 1) ** 2 * 4),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_rate),
                    )
                )

        self.fc_final = nn.Linear(hidden_dim * (num_layers + 1) ** 2 * 4, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent variable concatenated with condition embedding.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        x = torch.relu(self.fc_initial(z))
        x = self.dropout(x)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x_reconstructed = self.fc_final(x)

        return x_reconstructed


class ClassificationHead(torch.nn.Module):
    """Classification head for the Conditional Variational Autoencoder (CVAE).

    Maps latent variables to class logits.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            output_dim (int): Number of output classes.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
        """
        super(ClassificationHead, self).__init__()

        self.fc_initial = nn.Linear(latent_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Args:
            mu (torch.Tensor): Latent mean tensor.

        Returns:
            torch.Tensor: Logits for classification.
        """
        logits = self.fc_initial(mu)
        return logits


class MolecularEmbedding(torch.nn.Module):
    """Molecular embedding layer for the Conditional Variational Autoencoder (CVAE).

    Embeds condition (e.g., molecular fingerprint) data.
    """

    def __init__(
        self,
        condition_dim: int = 2048,
        condition_emb_dim: int = 64,
        num_layers: int = 0,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Args:
            condition_dim (int, optional): Dimension of the condition data. Defaults to 2048.
            condition_emb_dim (int, optional): Dimension of the embedded condition data. Defaults to 64.
            num_layers (int, optional): Number of hidden layers. Defaults to 0.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(MolecularEmbedding, self).__init__()
        self.fc_initial = nn.Linear(condition_dim, condition_emb_dim * (num_layers + 1))

        self.fc_layers = nn.ModuleList()
        if num_layers > 0:
            for layer_mult in range(1, num_layers + 1)[::-1]:
                self.fc_layers.append(
                    nn.Linear(
                        condition_emb_dim * (layer_mult + 1),
                        condition_emb_dim * layer_mult,
                    )
                )

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the molecular embedding layer.

        Args:
            c (torch.Tensor): Condition (fingerprint) tensor.

        Returns:
            torch.Tensor: Embedded condition tensor.
        """
        c = torch.relu(self.fc_initial(c))
        for fc_layer in self.fc_layers:
            c = torch.relu(fc_layer(c))
            c = self.dropout(c)
        return c


class CVAE(torch.nn.Module):
    """Conditional Variational Autoencoder for drug-induced gene expression prediction.

    Combines encoder, decoder, molecular embedding, and classification head.
    """

    def __init__(
        self,
        expression_dim: int,
        num_classes: int,
        num_encoder_layers: int = 0,
        encoder_dropout_rate: float = 0.2,
        num_decoder_layers: int = 0,
        decoder_dropout_rate: float = 0.2,
        condition_dim: int = 2048,
        condition_emb_dim: int = 64,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_molecular_emb_layers: int = 0,
    ) -> None:
        """
        Args:
            expression_dim (int): Dimension of the gene expression data.
            num_classes (int): Number of output classes.
            num_encoder_layers (int, optional): Number of encoder layers. Defaults to 0.
            encoder_dropout_rate (float, optional): Encoder dropout rate. Defaults to 0.2.
            num_decoder_layers (int, optional): Number of decoder layers. Defaults to 0.
            decoder_dropout_rate (float, optional): Decoder dropout rate. Defaults to 0.2.
            condition_dim (int, optional): Dimension of the condition data. Defaults to 2048.
            condition_emb_dim (int, optional): Dimension of the embedded condition data. Defaults to 64.
            hidden_dim (int, optional): Hidden dimension for encoder/decoder. Defaults to 512.
            latent_dim (int, optional): Dimension of the latent space. Defaults to 128.
            num_molecular_emb_layers (int, optional): Number of molecular embedding layers. Defaults to 0.
        """
        super(CVAE, self).__init__()
        self.expression_dim = expression_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.molecular_embedding = MolecularEmbedding(
            condition_dim=condition_dim,
            condition_emb_dim=condition_emb_dim,
            num_layers=num_molecular_emb_layers,
        )

        self.classification_head = ClassificationHead(
            latent_dim=latent_dim,
            output_dim=num_classes,
        )

        self.encoder = Encoder(
            input_dim=self.expression_dim + condition_emb_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=num_encoder_layers,
            dropout_rate=encoder_dropout_rate,
        )

        self.decoder = Decoder(
            input_dim=self.latent_dim + condition_emb_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.expression_dim,
            num_layers=num_decoder_layers,
            dropout_rate=decoder_dropout_rate,
        )

        self.use_checkpointing = True

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent space.

        Args:
            mu (torch.Tensor): Mean tensor.
            log_var (torch.Tensor): Log-variance tensor.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classify(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Classify the latent variable using the classification head.

        Args:
            mu (torch.Tensor): Latent mean tensor.

        Returns:
            torch.Tensor: Logits for classification.
        """
        logits = self.classification_head(mu)
        return logits

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CVAE.

        Args:
            x (torch.Tensor): Input gene expression tensor.
            c (torch.Tensor): Condition (fingerprint) tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed input, mean, and log-variance tensors.
        """

        if self.training and self.use_checkpointing:
            # Checkpoint the major networks to save memory
            c_embs = checkpoint(self.molecular_embedding, c, use_reentrant=False)
            x_cat = torch.cat((x, c_embs), dim=1)
            mu, logvar = checkpoint(self.encoder, x_cat, use_reentrant=False)
            z = self.reparameterize(mu, logvar)
            z_cat = torch.cat((z, c_embs), dim=1)
            x_reconstructed = checkpoint(self.decoder, z_cat, use_reentrant=False)

        else:
            c_embs = self.molecular_embedding(c)
            x = torch.cat((x, c_embs), dim=1)
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            z = torch.cat((z, c_embs), dim=1)
            x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, logvar
