import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GlobalAttention, global_max_pool
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. CONSTANTS (EXACT MATCH TO BENCHMARK)
# ==========================================
CSV_PATH = os.path.join('data', 'MDM2_Breaker', 'processed', 'benchmark_data.csv')
MDM2_SEQUENCE = "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN"
BATCH_SIZE = 128
LR = 0.001  # Matches DeepPurpose default
EPOCHS = 100

# ==========================================
# 2. THE MODEL (EXACT ARCHITECTURE MATCH)
# ==========================================
class DeepPurposeGCNLayer(nn.Module):
    """Exact replica of DeepPurpose's GCN Layer with Linear Residuals"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # DeepPurpose ALWAYS uses a Linear projection for residual
        self.res_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        return out + self.res_projection(x)

class GraphDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- DRUG ENCODER (DGL_GCN) ---
        # Layer 1: 74 -> 64
        self.gcn1 = DeepPurposeGCNLayer(74, 64)
        # Layer 2: 64 -> 64
        self.gcn2 = DeepPurposeGCNLayer(64, 64)
        # Layer 3: 64 -> 64
        self.gcn3 = DeepPurposeGCNLayer(64, 64)
        
        # Attention Pooling
        self.att_gate = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.att_gate)
        self.final_lin = nn.Linear(128, 256) # 64(sum) + 64(max) -> 256

        # --- PROTEIN ENCODER (CNN) ---
        # 1-hot (26 chars) -> 32 -> 64 -> 96
        self.prot_conv1 = nn.Conv1d(26, 32, kernel_size=4)
        self.prot_conv2 = nn.Conv1d(32, 64, kernel_size=8)
        self.prot_conv3 = nn.Conv1d(64, 96, kernel_size=12)
        self.prot_fc = nn.Linear(96, 256)

        # --- CLASSIFIER (MLP) ---
        # Input: 256 (Drug) + 256 (Prot) = 512
        # Hidden: [1024, 1024, 512] -> 1
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, protein_seq, molecule_data):
        # 1. Drug Path
        x, edge_index, batch = molecule_data.x.float(), molecule_data.edge_index, molecule_data.batch
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = self.gcn3(x, edge_index)
        
        # Pooling: Sum + Max
        x_sum = self.pool(x, batch)
        x_max = global_max_pool(x, batch)
        drug_vec = self.final_lin(torch.cat([x_sum, x_max], dim=1)) # [B, 256]

        # 2. Protein Path
        # Input: [B, SeqLen] (Integers) -> One Hot [B, 26, SeqLen]
        p = F.one_hot(protein_seq.long(), num_classes=26).permute(0, 2, 1).float()
        p = F.relu(self.prot_conv1(p))
        p = F.relu(self.prot_conv2(p))
        p = F.relu(self.prot_conv3(p))
        p = F.adaptive_max_pool1d(p, 1).squeeze(2) # Global Max Pool
        prot_vec = self.prot_fc(p) # [B, 256]

        # 3. Combine
        if prot_vec.shape[0] != drug_vec.shape[0]:
            prot_vec = prot_vec.expand(drug_vec.shape[0], -1)
            
        return self.mlp(torch.cat([drug_vec, prot_vec], dim=1))

# ==========================================
# 3. DATASET (EXACT LOGIC MATCH)
# ==========================================
class BenchmarkDataset(Dataset):
    def __init__(self, csv_path, split_name):
        super().__init__()
        # Load Raw CSV
        df = pd.read_csv(csv_path)
        # Filter by split (train/val/test)
        self.df = df[df['split'] == split_name].reset_index(drop=True)
        
        # Initialize DGL Featurizer
        self.featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        
        # Pre-process all graphs into memory (Speed!)
        self.data_list = []
        print(f"Processing {split_name} set...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            data = self._process_one(row['SMILES'], row['pIC50'])
            if data: self.data_list.append(data)

    def _process_one(self, smiles, label):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            # CRITICAL: Replicate sanitize_and_flatten
            Chem.RemoveStereochemistry(mol)
            
            # Featurize
            feats = self.featurizer(mol)['h'].float()
            adj = Chem.GetAdjacencyMatrix(mol)
            edge_index = torch.tensor(adj).nonzero().t().contiguous()
            
            return Data(x=feats, edge_index=edge_index, y=torch.tensor([label]).float())
        except:
            return None

    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

# ==========================================
# 4. LIGHTNING SYSTEM
# ==========================================
class SanityCheckSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GraphDTAModel()
        # Tokenize Protein Once
        vocab = "ACDEFGHIKLMNPQRSTVWY"
        char_map = {c: i+1 for i, c in enumerate(vocab)}
        indices = [char_map.get(c, 0) for c in MDM2_SEQUENCE]
        # Pad to 1000 (Standard DeepPurpose)
        indices += [0] * (1000 - len(indices))
        self.register_buffer("prot_seq", torch.tensor(indices).unsqueeze(0))

    def forward(self, mol_batch):
        return self.model(self.prot_seq, mol_batch)

    def training_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        loss = F.mse_loss(preds, batch.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        loss = F.mse_loss(preds, batch.y)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    pl.seed_everything(42)
    
    # 1. Load Data
    train_ds = BenchmarkDataset(CSV_PATH, 'train')
    val_ds = BenchmarkDataset(CSV_PATH, 'val')
    test_ds = BenchmarkDataset(CSV_PATH, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Train
    system = SanityCheckSystem()
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False
    )
    print("\nStarting Training...")
    trainer.fit(system, train_loader, val_loader)
    
    # 3. Final Eval
    print("\nEvaluating on Test Set...")
    system.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = system(batch).squeeze()
            all_preds.extend(preds.tolist())
            all_true.extend(batch.y.tolist())
            
    mse = mean_squared_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
    
    print("="*30)
    print(f"FINAL RESULTS (Apples-to-Apples)")
    print(f"MSE: {mse:.4f}")
    print(f"R2 : {r2:.4f}")
    print("="*30)