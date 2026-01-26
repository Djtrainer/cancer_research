import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    GlobalAttention,
    global_max_pool,
    global_add_pool,
    global_mean_pool
)

class AtomEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        # 1. Embeddings for Categorical Features
        self.z_embed = torch.nn.Embedding(120, embedding_dim // 2)
        self.hyb_embed = torch.nn.Embedding(6, embedding_dim // 4)
        self.chir_embed = torch.nn.Embedding(3, embedding_dim // 4)

        # 2. Linear "Lifter" for Continuous Features (6 input features)
        self.float_lin = torch.nn.Linear(6, embedding_dim)

        # Final Output Dimension = (32 + 16 + 16) + 64 = 128
        self.out_dim = embedding_dim + embedding_dim

    def forward(self, x):
        # Slice the input tensor inside the forward pass
        # x shape: [Num_Atoms, 9]
        
        # Categorical: First 3 columns (cast to int/long)
        x_cat = x[:, :3].long()
        z = self.z_embed(x_cat[:, 0]) 
        h = self.hyb_embed(x_cat[:, 1]) 
        c = self.chir_embed(x_cat[:, 2]) 
        cats = torch.cat([z, h, c], dim=1)

        # Continuous: Remaining 6 columns
        x_scalar = x[:, 3:].float()
        floats = self.float_lin(x_scalar)

        # Fuse
        return torch.cat([cats, floats], dim=1)


class UniversalGraphLayer(nn.Module):
    """
    A universal wrapper that enforces the DeepPurpose 'Style'
    (Conv -> BN -> ReLU -> Dropout + Residual) across different architectures.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        layer_type="GCN", 
        dropout=0.1, 
        heads=4,
        edge_dim=None # Passed from the model config (e.g., 4)
    ):
        super().__init__()
        self.layer_type = layer_type
        self.dropout_rate = dropout

        # 1. Define the Graph Convolution
        if layer_type == "GCN":
            self.conv = GCNConv(in_channels, out_channels)

        elif layer_type == "GAT":
            # GATv2 is strictly better than GAT (dynamic vs static attention)
            # We ensure the output dimension (heads * per_head_dim) equals out_channels
            assert out_channels % heads == 0, "out_channels must be divisible by heads"
            self.conv = GATv2Conv(
                in_channels, out_channels // heads, heads=heads, concat=True, edge_dim=edge_dim
            )

        elif layer_type == "GIN":
            # GIN requires an internal MLP
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.conv = GINConv(gin_mlp, train_eps=True)


        elif layer_type == "GINE":
            # GINE (Node + Edge features)
            # 1. Matches your MLP structure exactly
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels), 
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            # 2. Uses PyG's internal edge projection via `edge_dim`
            if edge_dim is None:
                raise ValueError("GINE requires 'edge_dim' to be set!")
                
            self.conv = GINEConv(gin_mlp, train_eps=True, edge_dim=edge_dim)

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # 2. Standardization Layers (Matches your DeepPurposeGCNLayer)
        self.bn = nn.BatchNorm1d(out_channels)
        self.res_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        
        # --- Run Convolution ---
        if self.layer_type == "GINE":
            if edge_attr is None:
                raise ValueError("GINE requires edge_attr input!")
            out = self.conv(x, edge_index, edge_attr=edge_attr)
            
        elif self.layer_type == "GAT" and edge_attr is not None:
            out = self.conv(x, edge_index, edge_attr=edge_attr)
            
        else:
            # GCN or GIN (ignore edges)
            out = self.conv(x, edge_index)
    
        out = self.bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # B. Residual Path (Projected)
        res = self.res_projection(x)

        return out + res


class ProteinEncoderUniversal(nn.Module):
    """
    The Graph-Based equivalent of ProteinEncoderCNN.
    Treats the protein as a graph (Nodes=Residues, Edges=Contacts).
    """
    def __init__(self, in_channels=33, hidden_dim=64, out_dim=256, layer_type='GCN', dropout=0.1):
        super().__init__()
        
        # Stack 3 Graph Layers (same depth as your Drug Encoder)
        self.layer1 = UniversalGraphLayer(in_channels, hidden_dim, layer_type, dropout)
        self.layer2 = UniversalGraphLayer(hidden_dim, hidden_dim, layer_type, dropout)
        self.layer3 = UniversalGraphLayer(hidden_dim, hidden_dim, layer_type, dropout)

        # Pooling Strategy: Sum + Max (Matches your snippet)
        # We don't use Attention Pooling here to keep it distinct from the Drug Encoder
        # and because protein graphs are often larger/noisier.
        self.final_lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, data):
        # data is a PyG Batch object for the Protein
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # 1. Graph Convolution
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)

        # 2. Pooling (Sum + Max)
        # v_sum = global_add_pool(x, batch)
        v_mean = global_mean_pool(x, batch)
        v_max = global_max_pool(x, batch)

        # 3. Project to 256
        v_cat = torch.cat([v_mean, v_max], dim=1)
        return self.final_lin(v_cat)


class DrugEncoderUniversal(nn.Module):
    def __init__(self, 
                 in_channels=74,     # Ignored if use_atom_embeddings=True
                 hidden_dim=64, 
                 out_dim=256, 
                 layer_type='GCN', 
                 dropout=0.1,
                 use_atom_embeddings=False, # NEW TOGGLE
                 atom_embedding_dim=64,      # Hyperparam for AtomEncoder
                 edge_dim=None  # <--- NEW ARGUMENT
                ):
        super().__init__()
        self.use_atom_embeddings = use_atom_embeddings

        # --- 1. Handle Input Encoding ---
        if use_atom_embeddings:
            # Use your custom AtomEncoder
            self.atom_encoder = AtomEncoder(embedding_dim=atom_embedding_dim)
            # The input size for the GCN is now the AtomEncoder's output size (e.g., 128)
            gnn_in_channels = self.atom_encoder.out_dim
        else:
            # Use raw features (like DeepPurpose's 74)
            self.atom_encoder = None
            gnn_in_channels = in_channels

        # --- 2. Stack Graph Layers ---
        self.layer1 = UniversalGraphLayer(gnn_in_channels, hidden_dim, layer_type, dropout, edge_dim=edge_dim)
        self.layer2 = UniversalGraphLayer(hidden_dim, hidden_dim, layer_type, dropout, edge_dim=edge_dim)
        self.layer3 = UniversalGraphLayer(hidden_dim, hidden_dim, layer_type, dropout, edge_dim=edge_dim)

        # --- 3. Pooling & Projection ---
        self.att_gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.attention_pool = GlobalAttention(gate_nn=self.att_gate_nn)
        self.final_lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        
        # A. Encode Atoms (if enabled)
        if self.use_atom_embeddings:
            # Pass the raw 9-column tensor; AtomEncoder handles slicing
            x = self.atom_encoder(x)
        else:
            # Standard DeepPurpose path (float cast needed)
            x = x.float()

        # B. Graph Convolution
        x = self.layer1(x, edge_index, edge_attr=edge_attr)
        x = self.layer2(x, edge_index, edge_attr=edge_attr)
        x = self.layer3(x, edge_index, edge_attr=edge_attr)

        # C. Pooling
        v_sum = self.attention_pool(x, batch)
        v_max = global_max_pool(x, batch)
        v_cat = torch.cat([v_sum, v_max], dim=1)
        
        return self.final_lin(v_cat)


class ProteinEncoderCNN(nn.Module):
    """
    Replicates the 'model_protein' (CNN) from DeepPurpose.
    Structure: One-Hot -> 3 Conv1D layers -> Global Max Pool -> Linear
    """

    def __init__(self, vocab_size=26, out_dim=256):
        super().__init__()
        self.vocab_size = vocab_size

        # Based on logs: Conv1d(26, 32...)
        # This implies the input is One-Hot Encoded (26 channels)
        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=12)

        self.fc1 = nn.Linear(96, out_dim)

    def forward(self, x):
        # x input is indices: [Batch, Seq_Len]
        # We need to One-Hot Encode it: [Batch, 26, Seq_Len]

        # Create one-hot (this matches the '26' input channel in the logs)
        x = F.one_hot(x.long(), num_classes=self.vocab_size)  # [Batch, Seq, 26]
        x = x.permute(0, 2, 1).float()  # [Batch, 26, Seq] for Conv1d

        # CNN Layers
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        # Global Max Pooling across the sequence dimension
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)  # [Batch, 96]

        # Final Projection
        x = self.fc1(x)
        return x


class GraphDTAModel(nn.Module):
    """
    The Full Siamese Network replicating DeepPurpose GraphDTA.
    """

    def __init__(
        self,
        molecule_in_channels=9,  # DEFAULT is 9 for your current data, DeepPurpose uses 74
        hidden_dim=256,  # Both encoders project to 256
        layer_type="GCN",
        use_atom_embeddings=False,
        atom_embedding_dim=64,
        drug_edge_dim=None,
        protein_mode="sequence",       # 'sequence' or 'graph'
        protein_vocab_size=26,         # For CNN (Sequence mode)
        protein_in_channels=33,        # For GCN (Graph mode) - Default Graphein features
        protein_layer_type="GCN",      # For GCN (Graph mode)
    ):
        super().__init__()
        
        self.protein_mode = protein_mode

        # 1. Encoders
        self.drug_encoder = DrugEncoderUniversal(
            in_channels=molecule_in_channels,
            hidden_dim=64,  # Internal GCN dim (from logs)
            out_dim=hidden_dim,
            layer_type=layer_type,
            use_atom_embeddings=use_atom_embeddings,
            atom_embedding_dim=atom_embedding_dim,
            edge_dim=drug_edge_dim
        )

        # 2. Protein Encoder (Switchable)
        if protein_mode == "graph":
            self.protein_encoder = ProteinEncoderUniversal(
                in_channels=protein_in_channels,
                hidden_dim=64,
                out_dim=hidden_dim,
                layer_type=protein_layer_type
            )
        else:
            # Default to Sequence CNN
            self.protein_encoder = ProteinEncoderCNN(
                vocab_size=protein_vocab_size, out_dim=hidden_dim
            )

        # 2. Predictor (MLP)
        # Input: 256 (Drug) + 256 (Protein) = 512
        # DeepPurpose Log: 512 -> 1024 -> 1024 -> 512 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Apply Xavier Uniform initialization to all Linear layers
        # This is the "Gold Standard" for these types of networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # Switch to Kaiming Initialization for ReLU networks
    #             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Conv1d):
    #             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, protein_input, molecule_data):
        """
        protein_input: [Batch, Seq_Len] (Integer indices)
        molecule_data: PyG Data Batch object
        """
        # Encode
        drug_vec = self.drug_encoder(molecule_data)  # [Batch, 256]
        prot_vec = self.protein_encoder(protein_input)  # [Batch, 256]

        # Expand Protein if needed (Broadcasting for 1 Protein vs Many Ligands)
        if prot_vec.shape[0] != drug_vec.shape[0]:
            prot_vec = prot_vec.expand(drug_vec.shape[0], -1)

        # Concat
        combined = torch.cat([drug_vec, prot_vec], dim=1)

        # Predict
        return self.mlp(combined)
