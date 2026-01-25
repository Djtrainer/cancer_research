import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,

)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = Path(os.getcwd()).parents[0]


class ScientificProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # 1. Get the standard metrics dict
        items = super().get_metrics(trainer, model)
        
        # 2. Find the 'lr' key and reformat it
        # (We check multiple keys because sometimes it's 'lr', 'lr-Adam', etc.)
        for key, value in items.items():
            if "lr" in key.lower() and isinstance(value, (int, float)):
                # Format: 1.23e-04
                items[key] = f"{value:.2e}"
                
        return items


class SarcomaScoutSystem(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, protein_data, lr: float = 0.00005):
        """
        Universal wrapper for Siamese Networks.
        Adapts automatically to Structure-Based (Graph) or Sequence-Based inputs.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "protein_data"])
        self.model = model
        self.lr = lr

        if isinstance(protein_data, Data):
            # It's a Graph (Structure-Based)
            self.mode = "graph"
            self.register_buffer("p_x", protein_data.x)
            self.register_buffer("p_edge_index", protein_data.edge_index)
            # Create dummy batch vector [0, 0, 0...]
            batch_vec = torch.zeros(protein_data.x.size(0), dtype=torch.long)
            self.register_buffer("p_batch", batch_vec)

        elif isinstance(protein_data, torch.Tensor):
            # It's a Sequence (Sequence-Based)
            self.mode = "sequence"
            # Ensure it is 2D: [1, Seq_Len]
            if protein_data.dim() == 1:
                protein_data = protein_data.unsqueeze(0)
            self.register_buffer("p_seq", protein_data)

        else:
            raise ValueError(f"Unknown protein data type: {type(protein_data)}")

    def get_protein_input(self):
        """Reconstructs the correct input object from GPU buffers."""
        if self.mode == "graph":
            return Data(x=self.p_x, edge_index=self.p_edge_index, batch=self.p_batch)
        else:
            # Return the sequence tensor directly
            return self.p_seq

    def forward(self, protein_input, molecule_data):
        # The model inside handles the specifics (e.g. expanding batch dims)
        return self.model(protein_input, molecule_data)

    def training_step(self, batch, batch_idx):
        # 1. Get the Static Protein (on the correct GPU)
        protein_input = self.get_protein_input()
        molecule_data = batch

        # 2. Predict
        preds = self(protein_input, molecule_data)

        # 3. Loss
        loss = F.mse_loss(preds.squeeze(), molecule_data.y)
        
        # --- NEW: Get and Log LR ---
        # Get the current LR from the optimizer
        lr = self.optimizers().param_groups[0]['lr']
        # Log it with prog_bar=True so it shows up in the console
        self.log("lr", lr, prog_bar=True, batch_size=batch.num_graphs)
        # ---------------------------

        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        protein_input = self.get_protein_input()
        molecule_data = batch
        preds = self(protein_input, molecule_data)
        loss = F.mse_loss(preds.squeeze(), molecule_data.y)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=1e-3,            # Peak LR (Match DeepPurpose's 0.001)
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=0.3,          # Spend 30% of time warming up
                    div_factor=50,          # Start at max_lr / 50
                    final_div_factor=10,  # End at max_lr / 10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update every batch, not every epoch
            },
        }


def define_trainer(
    epochs=100, lr=1e-4, save_name="sarcoma-model-{epoch:02d}-{val_loss:.2f}"
):
    # Checkpoint: Save the model with the lowest Validation Loss
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=save_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Early Stopping: Stop if val_loss doesn't improve for 10 epochs
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=100, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = ScientificProgressBar()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        log_every_n_steps=10,
    )

    return trainer


def generate_loaders(
    data_class,
    train_indices,
    val_indices,
    test_indices,
    batch_size=64,
    mol_file=None,
):
    if mol_file is None:
        raise ValueError("mol_file is required")

    data = data_class(
        root=os.path.join(ROOT, "data", "MDM2_Breaker"), file_path=mol_file
    )

    train_dataset = data[train_indices]
    val_dataset = data[val_indices]
    test_dataset = data[test_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def train_model(
    model,
    mol_class,
    protein_data,
    train_indices,
    val_indices,
    test_indices,
    epochs=100,
    lr=1e-4,
    save_name="sarcoma-model-{epoch:02d}-{val_loss:.2f}",
    train=True,
    mol_file=None,
    batch_size=64,
    seed=42,
):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    
    trainer = define_trainer(epochs=epochs, lr=lr, save_name=save_name)

    train_loader, val_loader, test_loader = generate_loaders(
        mol_class, train_indices, val_indices, test_indices, mol_file=mol_file, batch_size=batch_size
    )

    system = SarcomaScoutSystem(model=model, protein_data=protein_data, lr=lr)

    if train:
        trainer.fit(system, train_loader, val_loader)

    return model, train_loader, val_loader, test_loader


def load_model_from_checkpoint(model_class, checkpoint_path):
    # 1. Load the raw checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    # 2. Fix the keys (Remove "model." prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            # Strip the first 6 characters ("model.")
            new_key = key[6:]
            new_state_dict[new_key] = value

    # 3. Initialize your raw model
    model = model_class()

    # 4. Load the cleaned weights
    model.load_state_dict(new_state_dict)
    model.eval()

    return model
