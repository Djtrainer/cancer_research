import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, AttentionalAggregation, global_add_pool, global_max_pool, GATv2Conv
from torch_geometric.data import Data


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
        self.conv2 = GATv2Conv(out_channels * 8 * 4, out_channels * 4, heads=4, edge_dim=4)
        self.conv3 = GATv2Conv(out_channels * 4 * 4, out_channels * 2, heads=4, edge_dim=4)
        self.conv4 = GATv2Conv(out_channels * 2 * 4, out_channels, heads=4, concat=False, edge_dim=4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
        return x


class GraphSiameseNetwork(torch.nn.Module):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork, self).__init__()
        self.protein_encoder = GCNEncoder(
            in_channels=protein_in_channels, out_channels=out_channels
        )
        self.molecule_embedding = torch.nn.Embedding(
            molecule_in_channels, molecule_embedding_dim
        )
        self.molecule_encoder = GCNEncoder(
            in_channels=molecule_embedding_dim, out_channels=out_channels
        )
        # Calculate the attention weights for the embeddings
        self.attention_pool = AttentionalAggregation(gate_nn=torch.nn.Linear(out_channels, 1))
        
        hidden_channels = 4 * out_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 4, 1),
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

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        # Molecule Embedding
        # [total_mol_atoms, 1] -> [total_mol_atoms, molecule_embedding_dim]
        molecule_embedding = self.molecule_embedding(molecule_data.x.squeeze())
        # Molecule Encoding
        # [total_mol_atoms, molecule_embedding_dim] -> [total_mol_atoms, molecule_out_channels]
        molecule_embedding = self.molecule_encoder(molecule_embedding, molecule_data.edge_index)
        # Attention Pooling
        # [total_mol_atoms, molecule_out_channels] -> [batch_size, molecule_out_channels]
        # molecule_embedding = self.attention_pool(molecule_embedding, molecule_data.batch)
        # return molecule_embedding
        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)
        
        # Concatenate them (Output size: out_channels * 2)
        return torch.cat([v_sum, v_max], dim=1)

    def forward(self, protein_data: Data, molecule_data: Data) -> torch.Tensor:
        # Protein Encoding
        # [num_protein_nodes, protein_in_channels] -> [batch_size, protein_out_channels]
        protein_embedding = self.encode_protein(protein_data)

        # Molecule Encoding
        # [num_molecule_nodes, 1] -> [batch_size, molecule_out_channels]
        molecule_embedding = self.encode_molecule(molecule_data)
        
        # Combine the protein and molecule embeddings
        batch_size = molecule_embedding.size(0)
        protein_embedding_expanded = protein_embedding.expand(batch_size, -1)
        # [batch_size, protein_out_channels] + [batch_size, molecule_out_channels] -> 
        # [batch_size, protein_out_channels + molecule_out_channels]
        combined_embedding = torch.cat((protein_embedding_expanded, molecule_embedding), dim=1)

        # MLP
        # [batch_size, protein_out_channels + molecule_out_channels] -> [batch_size, 1]
        return self.mlp(combined_embedding)


class GraphSiameseNetwork_v2(torch.nn.Module):
    def __init__(
        self,
        protein_in_channels: int = 20,
        molecule_in_channels: int = 9,
        out_channels: int = 128,
        molecule_embedding_dim: int = 64,
    ):
        super(GraphSiameseNetwork_v2, self).__init__()
        self.protein_encoder = GCNEncoder(
            in_channels=protein_in_channels, out_channels=out_channels
        )
        self.molecule_embedding = torch.nn.Embedding(
            molecule_in_channels, molecule_embedding_dim
        )
        self.molecule_encoder = GatCNEncoder(
            in_channels=molecule_embedding_dim, out_channels=out_channels,
        )
        # Calculate the attention weights for the embeddings
        self.attention_pool = AttentionalAggregation(gate_nn=torch.nn.Linear(out_channels, 1))
        
        hidden_channels = 4 * out_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, hidden_channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 4, 1),
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

    def encode_molecule(self, molecule_data: Data) -> torch.Tensor:
        # Molecule Embedding
        # [total_mol_atoms, 1] -> [total_mol_atoms, molecule_embedding_dim]
        molecule_embedding = self.molecule_embedding(molecule_data.x.squeeze())
        # Molecule Encoding
        # [total_mol_atoms, molecule_embedding_dim] -> [total_mol_atoms, molecule_out_channels]
        molecule_embedding = self.molecule_encoder(molecule_embedding, molecule_data.edge_index, molecule_data.edge_attr)
        # Attention Pooling
        # [total_mol_atoms, molecule_out_channels] -> [batch_size, molecule_out_channels]
        # molecule_embedding = self.attention_pool(molecule_embedding, molecule_data.batch)
        # return molecule_embedding
        v_sum = global_add_pool(molecule_embedding, molecule_data.batch)
        v_max = global_max_pool(molecule_embedding, molecule_data.batch)
        
        # Concatenate them (Output size: out_channels * 2)
        return torch.cat([v_sum, v_max], dim=1)

    def forward(self, protein_data: Data, molecule_data: Data) -> torch.Tensor:
        # Protein Encoding
        # [num_protein_nodes, protein_in_channels] -> [batch_size, protein_out_channels]
        protein_embedding = self.encode_protein(protein_data)

        # Molecule Encoding
        # [num_molecule_nodes, 1] -> [batch_size, molecule_out_channels]
        molecule_embedding = self.encode_molecule(molecule_data)
        
        # Combine the protein and molecule embeddings
        batch_size = molecule_embedding.size(0)
        protein_embedding_expanded = protein_embedding.expand(batch_size, -1)
        # [batch_size, protein_out_channels] + [batch_size, molecule_out_channels] -> 
        # [batch_size, protein_out_channels + molecule_out_channels]
        combined_embedding = torch.cat((protein_embedding_expanded, molecule_embedding), dim=1)

        # MLP
        # [batch_size, protein_out_channels + molecule_out_channels] -> [batch_size, 1]
        return self.mlp(combined_embedding)
