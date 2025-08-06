from typing import Tuple

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset



class LincsDataset(Dataset):
    """Dataset for LINCS drug-induced gene expression prediction."""

    def __init__(self, df_expression: pd.DataFrame, df_meta: pd.DataFrame):
        """
        Args:
            df_expression (pd.DataFrame): DataFrame containing gene expression data with genes as columns and samples pert_ids as rows.
            fingerprint_map (Dict[str, np.ndarray]): Dictionary mapping perturbation IDs to their corresponding fingerprints.
        """
        self.df_expression = df_expression
        self.df_meta = df_meta
        # Ensure that the perturbation IDs in the expression data match those in the metadata
        self.pert_ids = df_expression.index.intersection(self.df_meta.index).tolist()

        self.df_expression = df_expression.loc[self.pert_ids]
        self.df_meta = df_meta.loc[self.pert_ids]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.df_expression)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the fingerprint and gene expression values for a given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the fingerprint and the gene expression values.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds for dataset.")

        pert_id = self.pert_ids[idx]
        if pert_id not in self.df_meta.index:
            raise KeyError(f"Perturbation ID {pert_id} not found in metadata.")

        # Convert to PyTorch tensors
        expression_tensor = torch.tensor(
            self.df_expression.loc[pert_id].values, dtype=torch.float
        )
        fingerprint_tensor = torch.tensor(
            self.df_meta.loc[pert_id, "fingerprint"], dtype=torch.float
        )
        moa_label_tensor = torch.tensor(
            self.df_meta.loc[pert_id, "moa_int"], dtype=torch.int64
        )

        return fingerprint_tensor, expression_tensor, moa_label_tensor

    def get_item_by_pert_id(self, pert_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the fingerprint and gene expression values for a given perturbation ID.
        Args:
            pert_id (str): Perturbation ID to retrieve.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the fingerprint and the gene expression values.
        """
        if pert_id not in self.df_meta.index:
            raise KeyError(f"Perturbation ID {pert_id} not found in metadata.")

        expression_tensor = torch.tensor(
            self.df_expression.loc[pert_id].values, dtype=torch.float
        )
        fingerprint_tensor = torch.tensor(
            self.df_meta.loc[pert_id, "fingerprint"], dtype=torch.float
        )
        moa_label_tensor = torch.tensor(
            self.df_meta.loc[pert_id, "moa_int"], dtype=torch.int64
        )

        return fingerprint_tensor, expression_tensor, moa_label_tensor
