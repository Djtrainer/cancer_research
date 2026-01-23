import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GlobalAttention, global_max_pool


class DeepPurposeGCNLayer(nn.Module):
    """
    Replicates the 'GCNLayer' from DeepPurpose/DGL.
    Logic: Output = ReLU(BN(GCN(x))) + Residual(x)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # FIX: DeepPurpose ALWAYS uses a Linear projection for the residual,
        # even if dimensions are the same. This adds learnable parameters.
        self.res_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 1. Main Path
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        
        # 2. Residual Path (Projected)
        res = self.res_projection(x)
        
        return out + res


class DrugEncoderGCN(nn.Module):
    """
    Replicates the 'model_drug' (DGL_GCN) from DeepPurpose.
    Structure: 3 GCN Layers -> Attention+Max Pooling -> Linear Projection
    """

    def __init__(self, in_channels=74, hidden_dim=64, out_dim=256):
        super().__init__()

        # 3 Stacked GCN Layers
        # Note: DeepPurpose typically does Input->64->64->64
        self.layer1 = DeepPurposeGCNLayer(in_channels, hidden_dim)
        self.layer2 = DeepPurposeGCNLayer(hidden_dim, hidden_dim)
        self.layer3 = DeepPurposeGCNLayer(hidden_dim, hidden_dim)

        # Attention Pooling Mechanism (WeightedSumAndMax)
        # We need a small NN to calculate attention weights
        self.att_gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.attention_pool = GlobalAttention(gate_nn=self.att_gate_nn)

        # Final projection: (Sum_Dim + Max_Dim) -> Output
        # Since we concat Sum + Max, input is hidden_dim * 2
        self.final_lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # 1. GCN Layers
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)

        # 2. Pooling (Weighted Sum + Max)
        # GlobalAttention performs the weighted sum
        v_sum = self.attention_pool(x, batch)
        # global_max_pool performs the max
        v_max = global_max_pool(x, batch)

        # 3. Concatenate and Project
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
        protein_vocab_size=26,
        hidden_dim=256,  # Both encoders project to 256
    ):
        super().__init__()

        # 1. Encoders
        self.drug_encoder = DrugEncoderGCN(
            in_channels=molecule_in_channels,
            hidden_dim=64,  # Internal GCN dim (from logs)
            out_dim=hidden_dim,
        )

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
