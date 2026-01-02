import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels * 8)
        self.conv2 = GCNConv(out_channels * 8, out_channels * 4)
        self.conv3 = GCNConv(out_channels * 4, out_channels * 2)
        self.conv4 = GCNConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x


class GraphSiameseNetwork(torch.nn.Module):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        protein_out_channels: int = 128,
        molecule_out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork, self).__init__()
        self.protein_encoder = GCNEncoder(
            in_channels=protein_in_channels, out_channels=protein_out_channels
        )
        self.molecule_embedding = torch.nn.Embedding(
            molecule_in_channels, molecule_embedding_dim
        )
        self.molecule_encoder = GCNEncoder(
            in_channels=molecule_embedding_dim, out_channels=molecule_out_channels
        )

        hidden_channels = protein_out_channels + molecule_out_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 4, 1),
        )

    def forward(
        self, protein_data: torch.Tensor, molecule_data: torch.Tensor
    ) -> torch.Tensor:
        # Protein Encoding
        # [num_protein_nodes, protein_in_channels] -> [num_protein_nodes, protein_out_channels]
        protein_embedding = self.protein_encoder(
            protein_data.x, protein_data.edge_index
        )
        # Global Mean Pooling
        # [num_protein_nodes, protein_out_channels] -> [batch_size, protein_out_channels]
        protein_embedding = global_mean_pool(protein_embedding, protein_data.batch)

        # Molecule Embedding
        # [total_mol_atoms, 1] -> [total_mol_atoms, molecule_embedding_dim]
        molecule_embedding = self.molecule_embedding(molecule_data.x.squeeze())
        # Molecule Encoding
        # [total_mol_atoms, molecule_embedding_dim] -> [total_mol_atoms, molecule_out_channels]
        molecule_embedding = self.molecule_encoder(
            molecule_embedding, molecule_data.edge_index
        )
        # Global Mean Pooling
        # [total_mol_atoms, molecule_out_channels] -> [batch_size, molecule_out_channels]
        molecule_embedding = global_mean_pool(molecule_embedding, molecule_data.batch)
        
        
        # Combine the protein and molecule embeddings
        batch_size = molecule_embedding.size(0)
        protein_embedding_expanded = protein_embedding.expand(batch_size, -1)
        # [batch_size, protein_out_channels] + [batch_size, molecule_out_channels] -> 
        # [batch_size, protein_out_channels + molecule_out_channels]
        combined_embedding = torch.cat((protein_embedding_expanded, molecule_embedding), dim=1)

        # MLP
        # [batch_size, protein_out_channels + molecule_out_channels] -> [batch_size, 1]
        return self.mlp(combined_embedding)
