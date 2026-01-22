import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINEConv,
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

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
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

    def forward_with_attention(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass through first 3 layers normally
        x, (att_edge_index, att_weights) = self.conv1(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)

        return x, att_edge_index, att_weights


class GINEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Helper to create the MLP required by GINE
        # Input -> Batch Norm -> ReLU -> Linear
        def make_mlp(in_dim, out_dim):
            return torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.BatchNorm1d(out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, out_dim),
            )

        # Layer 1: Project from Input (128) to Output (128)
        self.conv1 = GINEConv(make_mlp(in_channels, out_channels), edge_dim=4)

        # Layer 2: Keep dims constant (128 -> 128)
        self.conv2 = GINEConv(make_mlp(out_channels, out_channels), edge_dim=4)

        # Layer 3
        self.conv3 = GINEConv(make_mlp(out_channels, out_channels), edge_dim=4)

        # Layer 4
        self.conv4 = GINEConv(make_mlp(out_channels, out_channels), edge_dim=4)

    def forward(self, x, edge_index, edge_attr):
        # GINE includes ReLU inside the MLP, so we often don't need extra ReLUs here,
        # but adding them between blocks is fine.
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)
        return x


class AtomEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        # 1. Embeddings for Categorical Features
        # Atomic Num (0-118)
        self.z_embed = torch.nn.Embedding(120, embedding_dim // 2)
        # Hybridization (0-5)
        self.hyb_embed = torch.nn.Embedding(6, embedding_dim // 4)
        # Chirality (0-2)
        self.chir_embed = torch.nn.Embedding(3, embedding_dim // 4)

        # 2. Linear "Lifter" for Continuous Features
        # Input: 6 floats (Mass, Degree, Charge, Aromatic, Ring, H-Count)
        # Output: Match embedding dimension
        self.float_lin = torch.nn.Linear(6, embedding_dim)

        # Final Output Dimension = (32 + 16 + 16) + 64 = 128
        self.out_dim = embedding_dim + embedding_dim

    def forward(self, x_cat, x_scalar):
        # x_cat shape: [Num_Atoms, 3]
        z = self.z_embed(x_cat[:, 0])  # [N, 32]
        h = self.hyb_embed(x_cat[:, 1])  # [N, 16]
        c = self.chir_embed(x_cat[:, 2])  # [N, 16]

        # Concatenate embeddings -> [N, 64]
        cats = torch.cat([z, h, c], dim=1)

        # Project floats -> [N, 64]
        floats = self.float_lin(x_scalar)

        # Fuse them -> [N, 128]
        return torch.cat([cats, floats], dim=1)


class GraphSiameseNetwork(torch.nn.Module):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
        fingerprints_model_dim: int = 2052,
        molecular_encoding_type: str = "gin",
    ):
        super(GraphSiameseNetwork, self).__init__()
        self.protein_encoder = GCNEncoder(
            in_channels=protein_in_channels, out_channels=out_channels
        )
        self.atom_encoder = AtomEncoder(embedding_dim=molecule_embedding_dim)
        gnn_input_dim = self.atom_encoder.out_dim

        if molecular_encoding_type == "gin":
            self.molecule_encoder = GINEEncoder(
                in_channels=gnn_input_dim,
                out_channels=out_channels,
            )
        elif molecular_encoding_type == "gcn":
            self.molecule_encoder = GCNEncoder(
                in_channels=gnn_input_dim,
                out_channels=out_channels,
            )
        elif molecular_encoding_type == "gat":
            self.molecule_encoder = GatCNEncoder(
                in_channels=gnn_input_dim,
                out_channels=out_channels,
            )
        else:
            raise ValueError(
                f"Invalid molecular encoding type: {molecular_encoding_type}"
            )

        gnn_output_dim = out_channels * 4
        total_input_dim = gnn_output_dim
        # total_input_dim = gnn_output_dim + fingerprints_model_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1),
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

    def encode_molecule_inner(
        self,
        x_embed: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Internal helper: Runs GNN + Pooling on pre-computed embeddings.
        """
        # 1. GINE Convolutions
        x = self.molecule_encoder(x_embed, edge_index, edge_attr)

        # 2. Pooling
        v_sum = global_add_pool(x, batch)
        v_max = global_max_pool(x, batch)

        return torch.cat([v_sum, v_max], dim=1)

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        # 1. Atom Encoder
        x_cat = molecule_data.x[:, :3].long()
        x_scalar = molecule_data.x[:, 3:].float()
        molecule_embedding = self.atom_encoder(x_cat, x_scalar)

        # 2. Inner GNN + Pooling
        return self.encode_molecule_inner(
            molecule_embedding,
            molecule_data.edge_index,
            molecule_data.edge_attr,
            molecule_data.batch,
        )

    def forward(self, protein_data: Data, molecule_data: Data) -> torch.Tensor:
        # Standard forward pass
        prot_vec = self.encode_protein(protein_data)
        mol_vec = self.encode_molecule(molecule_data)

        if prot_vec.shape[0] != mol_vec.shape[0]:
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        # expert_features = molecule_data.expert_features.squeeze(1)
        # combined = torch.cat([prot_vec, mol_vec, expert_features], dim=1)
        combined = torch.cat([prot_vec, mol_vec], dim=1)
        return self.mlp(combined)

    def forward_from_embeddings(
        self, mol_embed: torch.Tensor, molecule_data: Data, protein_data: Data
    ) -> torch.Tensor:
        """
        Special Forward Pass for Gradient/Design Scripts.
        Takes PRE-COMPUTED molecule embeddings (with gradients enabled)
        and runs the rest of the network.
        """
        # 1. Run Molecule GNN (using the hooked embeddings)
        mol_vec = self.encode_molecule_inner(
            mol_embed,
            molecule_data.edge_index,
            molecule_data.edge_attr,
            molecule_data.batch,
        )

        # 2. Run Protein GNN (Standard)
        prot_vec = self.encode_protein(protein_data)
        if prot_vec.shape[0] != mol_vec.shape[0]:
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        # 3. Expert Features & MLP
        expert_features = molecule_data.expert_features.squeeze(1)
        combined = torch.cat([prot_vec, mol_vec, expert_features], dim=1)
        return self.mlp(combined)
