from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    global_add_pool,
    global_max_pool,
)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels * 8)
        self.conv2 = GCNConv(out_channels * 8, out_channels * 4)
        self.conv3 = GCNConv(out_channels * 4, out_channels * 2)
        self.conv4 = GCNConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x


class GatCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GatCNEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels * 8, heads=4, edge_dim=4)
        self.conv2 = GATv2Conv(
            out_channels * 8 * 4, out_channels * 4, heads=4, edge_dim=4
        )
        self.conv3 = GATv2Conv(
            out_channels * 4 * 4, out_channels * 2, heads=4, edge_dim=4
        )
        self.conv4 = GATv2Conv(
            out_channels * 2 * 4, out_channels, heads=4, concat=False, edge_dim=4
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
        return x


class GraphSiameseNetworkBase(torch.nn.Module, ABC):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetworkBase, self).__init__()
        self.protein_encoder = GCNEncoder(
            in_channels=protein_in_channels, out_channels=out_channels
        )
        self.hidden_channels = 4 * out_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels // 2, self.hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_channels // 4, 1),
        )

    def encode_protein(self, protein_data: Data) -> torch.Tensor:
        # Protein Encoding
        # [num_protein_nodes, protein_in_channels] -> [num_protein_nodes, protein_out_channels]
        protein_embedding = self.protein_encoder(
            protein_data.x, protein_data.edge_index
        )

        # Attention Pooling
        # [batch_size, protein_out_channels] -> [batch_size, protein_out_channels]
        # protein_embedding = self.attention_pool(protein_embedding, protein_data.batch)
        v_sum = global_add_pool(protein_embedding, protein_data.batch)
        v_max = global_max_pool(protein_embedding, protein_data.batch)

        return torch.cat([v_sum, v_max], dim=1)

    @abstractmethod
    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        """Child classes MUST define how to encode the molecule."""

    def forward(self, protein_data: Data, molecule_data: Data) -> torch.Tensor:
        """Shared Forward Pass."""
        # 1. Get Embeddings
        prot_vec = self.encode_protein(protein_data)  # [Batch, 2*Out]
        mol_vec = self.encode_molecule(molecule_data)  # [Batch, 2*Out]

        # 2. Expand Protein to match Batch Size (if needed)
        if prot_vec.shape[0] != mol_vec.shape[0]:
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        # 3. Concatenate & Predict
        combined = torch.cat([prot_vec, mol_vec], dim=1)
        return self.mlp(combined)


# --- VERSION 1 (GCN Encoder, Simple Atoms, No Bond Types) ---
class GraphSiameseNetwork_v1(GraphSiameseNetworkBase):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork_v1, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
        )

        self.molecule_embedding = torch.nn.Embedding(
            molecule_in_channels, molecule_embedding_dim
        )

        self.molecule_encoder = GCNEncoder(
            in_channels=molecule_embedding_dim, out_channels=out_channels
        )

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        # Molecule Embedding
        # [total_mol_atoms, 1] -> [total_mol_atoms, molecule_embedding_dim]
        molecule_embedding = self.molecule_embedding(molecule_data.x.squeeze())
        # Molecule Encoding
        # [total_mol_atoms, molecule_embedding_dim] -> [total_mol_atoms, molecule_out_channels]
        molecule_embedding = self.molecule_encoder(
            molecule_embedding, molecule_data.edge_index
        )

        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)

        # Concatenate them (Output size: out_channels * 2)
        return torch.cat([v_sum, v_max], dim=1)


# --- VERSION 2 (GAT Encoder, Simple Atoms + Bond Attributes) ---
class GraphSiameseNetwork_v2(GraphSiameseNetworkBase):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork_v2, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
        )
        self.molecule_embedding = torch.nn.Embedding(
            molecule_in_channels, molecule_embedding_dim
        )
        self.molecule_encoder = GatCNEncoder(
            in_channels=molecule_embedding_dim,
            out_channels=out_channels,
        )

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        # Molecule Embedding
        # [total_mol_atoms, 1] -> [total_mol_atoms, molecule_embedding_dim]
        molecule_embedding = self.molecule_embedding(molecule_data.x.squeeze())
        # Molecule Encoding
        # [total_mol_atoms, molecule_embedding_dim] -> [total_mol_atoms, molecule_out_channels]
        molecule_embedding = self.molecule_encoder(
            molecule_embedding, molecule_data.edge_index, molecule_data.edge_attr
        )

        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)

        # Concatenate them (Output size: out_channels * 2)
        return torch.cat([v_sum, v_max], dim=1)


# --- VERSION 3 (GAT Encoder, Rich Atoms + Bond Attributes) ---
class GraphSiameseNetwork_v3(GraphSiameseNetworkBase):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 8,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork_v3, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
        )

        self.molecule_encoder = GatCNEncoder(
            in_channels=molecule_in_channels,
            out_channels=out_channels,
        )

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        molecule_embedding = self.molecule_encoder(
            molecule_data.x.float(), molecule_data.edge_index, molecule_data.edge_attr
        )

        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)

        return torch.cat([v_sum, v_max], dim=1)
