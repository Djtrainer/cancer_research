from abc import ABC, abstractmethod

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
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
        molecule_in_channels: int = 128,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork_v3, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
        )
        self.atom_encoder = AtomEncoder(embedding_dim=molecule_embedding_dim)
        self.molecule_encoder = GatCNEncoder(
            in_channels=molecule_in_channels,
            out_channels=out_channels,
        )

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        x_cat = molecule_data.x[:, :3].long()
        x_scalar = molecule_data.x[:, 3:].float()

        molecule_embedding = self.atom_encoder(x_cat, x_scalar)
        molecule_embedding = self.molecule_encoder(
            molecule_embedding, molecule_data.edge_index, molecule_data.edge_attr
        )

        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)

        return torch.cat([v_sum, v_max], dim=1)

    def get_attention_for_molecule(
        self, molecule_data: Data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cat = molecule_data.x[:, :3].long()
        x_scalar = molecule_data.x[:, 3:].float()

        molecule_embedding = self.atom_encoder(x_cat, x_scalar)
        molecule_embedding, att_edge_index, att_weights = (
            self.molecule_encoder.forward_with_attention(
                molecule_embedding, molecule_data.edge_index, molecule_data.edge_attr
            )
        )

        return molecule_embedding, att_edge_index, att_weights


# --- VERSION 4 (GINEConv Encoder, Rich Atoms + Bond Attributes) ---
class GraphSiameseNetwork_v4(GraphSiameseNetworkBase):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 128,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
        molecular_encoding_type: str = 'gin',
    ):
        super(GraphSiameseNetwork_v4, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
        )
        self.atom_encoder = AtomEncoder(embedding_dim=molecule_embedding_dim)
        if molecular_encoding_type == 'gin':
            self.molecule_encoder = GINEEncoder(
                in_channels=molecule_in_channels,
                out_channels=out_channels,
            )
        elif molecular_encoding_type == 'gcn':
            self.molecule_encoder = GCNEncoder(
                in_channels=molecule_in_channels,
                out_channels=out_channels,
            )
        elif molecular_encoding_type == 'gat':
            self.molecule_encoder = GatCNEncoder(
                in_channels=molecule_in_channels,
                out_channels=out_channels,
            )
        else:
            raise ValueError(f"Invalid molecular encoding type: {molecular_encoding_type}")
    
    def encode_molecule_inner(
        self, 
        x_embed: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        batch: torch.Tensor
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
            molecule_data.batch
        )


# --- VERSION 5 (GINEConv Encoder, Rich Atoms + Bond Attributes) + Fingerprints Model ---
class GraphSiameseNetwork_v5(GraphSiameseNetwork_v4):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 128,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
        fingerprints_model_dim: int = 2052,
        molecular_encoding_type: str = 'gin',
    ):
        super(GraphSiameseNetwork_v5, self).__init__(
            protein_in_channels=protein_in_channels,
            molecule_in_channels=molecule_in_channels,
            out_channels=out_channels,
            molecule_embedding_dim=molecule_embedding_dim,
            molecular_encoding_type=molecular_encoding_type,
        )

        gnn_output_dim = out_channels * 4
        total_input_dim = gnn_output_dim + fingerprints_model_dim 
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1)
        )

    def forward(self, protein_data: Data, molecule_data: Data) -> torch.Tensor:
        # Standard forward pass
        prot_vec = self.encode_protein(protein_data)
        mol_vec = self.encode_molecule(molecule_data)

        if prot_vec.shape[0] != mol_vec.shape[0]:
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        expert_features = molecule_data.expert_features.squeeze(1)
        combined = torch.cat([prot_vec, mol_vec, expert_features], dim=1)
        return self.mlp(combined)

    def forward_from_embeddings(
        self, 
        mol_embed: torch.Tensor, 
        molecule_data: Data, 
        protein_data: Data
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
            molecule_data.batch
        )
        
        # 2. Run Protein GNN (Standard)
        prot_vec = self.encode_protein(protein_data)
        if prot_vec.shape[0] != mol_vec.shape[0]:
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        # 3. Expert Features & MLP
        expert_features = molecule_data.expert_features.squeeze(1)
        combined = torch.cat([prot_vec, mol_vec, expert_features], dim=1)
        return self.mlp(combined)


# --- GraphDTA-style 1D CNN for Protein Sequences ---
class ProteinSequenceEncoder(torch.nn.Module):
    def __init__(self, vocab_size=26, out_dim=256):
        super().__init__()
        self.vocab_size = vocab_size

        # DeepPurpose Architecture: 26 -> 32 -> 64 -> 96
        self.conv1 = torch.nn.Conv1d(in_channels=vocab_size, out_channels=32, kernel_size=4)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=12)

        self.fc1 = torch.nn.Linear(96, out_dim)

    def forward(self, x):
        # One-Hot Encode on the fly to avoid massive embeddings for single-target data
        x = F.one_hot(x.long(), num_classes=self.vocab_size).float()
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.fc1(x)


# --- GraphDTA-style Siamese Network 
# (GINEConv Encoder, Rich Atoms + Bond Attributes) + Protein Sequence Encoder ---
class SequenceSiameseNetwork(torch.nn.Module):
    def __init__(
        self, 
        molecule_encoder: torch.nn.Module,   
        sequence_encoder: torch.nn.Module,   
        fingerprints_dim: int = 2052,
        molecule_out_dim: int = 256,   
        protein_out_dim: int = 128,    
        mlp_hidden_dim: int = 1024
    ):
        super().__init__()
        
        # 1. The Components
        self.molecule_encoder = molecule_encoder
        self.protein_encoder = sequence_encoder
        
        # 2. The MLP (Replicating v5 architecture exactly for fairness)
        # Input = Protein_Vec + Molecule_Vec + Fingerprints
        # total_input_dim = protein_out_dim + molecule_out_dim + fingerprints_dim   
        total_input_dim = protein_out_dim + molecule_out_dim
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, mlp_hidden_dim),
            torch.nn.BatchNorm1d(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(mlp_hidden_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1)
        )

    def forward(self, protein_indices, molecule_data):
        """
        protein_indices: [Batch, Seq_Len] (Integer Tensor)
        molecule_data: PyG Data Object
        """
        
        # 1. Encode Protein (Sequence)
        # Output: [Batch, 128]
        prot_vec = self.protein_encoder(protein_indices)
        
        # 2. Encode Molecule (Graph) - Using your existing v5 logic
        # Output: [Batch, 256] (Sum + Max pool)
        mol_vec = self.molecule_encoder(molecule_data) 
        
        # 3. Handle Batch Expansion (if using Siamese pairs)
        if prot_vec.shape[0] != mol_vec.shape[0]:
             # Assuming 1 Protein vs Many Ligands (Broadcasting)
            prot_vec = prot_vec.expand(mol_vec.shape[0], -1)

        # 4. Expert Features (Fingerprints) - Critical to keep v5 parity
        # expert_features = molecule_data.expert_features.squeeze(1)
        
        # 5. Combine & Predict
        # combined = torch.cat([prot_vec, mol_vec, expert_features], dim=1)
        combined = torch.cat([prot_vec, mol_vec], dim=1)
        return self.mlp(combined)