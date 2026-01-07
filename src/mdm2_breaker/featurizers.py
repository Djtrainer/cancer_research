import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from abc import abstractmethod

from abc import ABC, abstractmethod

class ProteinFeaturizer:
    def __init__(self, pdb_file: str, contact_threshold=8):
        """
        Featurizes a protein structure into a PyG Data object.
        Args:
            pdb_file (str): Path to the PDB file.
            contact_threshold (int): Threshold for contact distance in Ångstroms.
        """
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.contact_threshold = contact_threshold

        # Define the 20 standard amino acids to one-hot encode them
        self.aa_codes = {
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLN": 5,
            "GLU": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
        }

    def _parse_structure(self):
        """
        Parses the protein structure into a PyG Data object.
        """
        structure = self.parser.get_structure("MDM2", self.pdb_file)

        coords = []
        features = []
        model = structure[0]
        for chain in model:
            for residue in chain:
                # Filter: Must be a standard amino acid (ignore water/HOH)
                if residue.get_resname() not in self.aa_codes:
                    continue

                # Filter: Must have an Alpha Carbon (CA)
                if "CA" in residue:
                    # Get positions
                    atom = residue["CA"]
                    coords.append(atom.get_coord())

                    # Get biological feature
                    res_name = residue.get_resname()
                    features.append(self.aa_codes[res_name])

        return np.array(coords), np.array(features)

    def _get_adjacency(self, coords: np.ndarray) -> torch.Tensor:
        """
        Computes the pairwise distance matrix and thresholds it to find edges.
        Args:
            coords (np.ndarray): Array of coordinates for the protein structure.
        Returns:
            torch.Tensor: Edge index tensor.
        """
        # Calculate the Euclidean distance between all nodes
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        # Create Edge list
        src, dst = np.where((distances < self.contact_threshold) & (distances > 0))

        # Convert to PyTorch tensor
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        return edge_index

    def get_graph(self):
        """
        Gets the protein structure as a PyG Data object.
        """
        # Parse
        coords, aa_indices = self._parse_structure()

        # Featurize Nodes
        x = torch.tensor(aa_indices, dtype=torch.long)
        x = torch.nn.functional.one_hot(x, num_classes=len(self.aa_codes)).float()

        # Compute Edges
        edge_index = self._get_adjacency(coords)

        # Pack into a Graph
        data = Data(
            x=x, edge_index=edge_index, pos=torch.tensor(coords, dtype=torch.float)
        )

        return data


class SmallMoleculeFeaturizerBase(InMemoryDataset, ABC):
    def __init__(self, root, file_path: str = None, transform=None, pre_transform=None):
        self.file_path = file_path
        super().__init__(root, transform, pre_transform)
        # Load the specific processed file for the child class
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """
        PyG expects a value here, but we don't need it.
        """
        return []

    @property
    @abstractmethod
    def processed_file_names(self):
        """Child classes must define their own filename to prevent collisions."""
        return ["mdm2_graphs.pt"]

    def download(self):
        """
        Download the dataset from the internet. (Not needed for this dataset)
        """
        pass

    def _load_and_clean_df(self):
        """
        Shared logic to load and clean the DataFrame.
        Available to all child classes.
        """
        df = pd.read_csv(self.file_path, sep="\t", on_bad_lines="skip", low_memory=False)

        # Select columns
        df = df[["Ligand SMILES", "IC50 (nM)", "Target Name", "Target Source Organism According to Curator or DataSource"]]
        df.columns = ["SMILES", "IC50", "Target", "Organism"]

        # Filter MDM2 + Human
        df = df[
            df["Target"].astype(str).str.contains("Mdm2", case=False)
            & df["Organism"].astype(str).str.contains("Homo sapiens", case=False)
        ]

        # Calculate pIC50
        df["IC50_numeric"] = pd.to_numeric(
            df["IC50"].astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce"
        )
        df["pIC50"] = -np.log10(df["IC50_numeric"] * 1e-9)
        df = df.dropna(subset=["pIC50", "SMILES"])

        # Normalize
        global_mean = df["pIC50"].mean()
        global_std = df["pIC50"].std()
        df["pIC50_norm"] = (df["pIC50"] - global_mean) / global_std
        
        return df, global_mean, global_std

    def _save_data(self, data_list, df, global_mean, global_std):
        """
        Shared logic to save data, stats, and the sidecar CSV.
        """
        if len(data_list) == 0:
            raise RuntimeError("No valid molecules found.")

        # Save Sidecar CSV
        csv_path = self.processed_paths[0].replace(".pt", ".csv")
        df.to_csv(csv_path, index=False)

        # Save PyG Data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Save Stats
        torch.save(
            {"mean": global_mean, "std": global_std},
            self.processed_paths[0].replace(".pt", "_stats.pt"),
        )
        print(f"Saved {self.processed_file_names[0]}")
        
    @abstractmethod
    def process(self):
        pass    


# --- VERSION 1 (Simple Atoms, No Bond Types) ---
class SmallMoleculeFeaturizer(SmallMoleculeFeaturizerBase):
    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v1.pt"]  

    def process(self):
        df, mean, std = self._load_and_clean_df()
        data_list = []
        allowed_atoms = [6, 7, 8, 9, 16, 17, 35, 53] 

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V1"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None: continue

            # V1 Node Features
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                node_feats.append(allowed_atoms.index(an) if an in allowed_atoms else len(allowed_atoms))
            
            x = torch.tensor(node_feats, dtype=torch.long).unsqueeze(1)

            # V1 Edge Index (Connectivity Only)
            row_idx, col_idx = [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row_idx += [start, end]
                col_idx += [end, start]

            data = Data(
                x=x, 
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float)
            )
            data_list.append(data)

        self._save_data(data_list, df, mean, std)


# --- VERSION 2 (Simple Atoms + Bond Attributes) ---
class SmallMoleculeFeaturizer_v2(SmallMoleculeFeaturizerBase):
    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v2.pt"] # 

    def process(self):
        df, mean, std = self._load_and_clean_df()
        data_list = []
        allowed_atoms = [6, 7, 8, 9, 16, 17, 35, 53]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V2"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None: continue

            # Nodes (Same as V1)
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                node_feats.append(allowed_atoms.index(an) if an in allowed_atoms else len(allowed_atoms))
            x = torch.tensor(node_feats, dtype=torch.long).unsqueeze(1)

            # Edges WITH Attributes
            row_idx, col_idx, bond_feats = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = self._get_bond_feature(bond)
                
                row_idx += [start, end]
                col_idx += [end, start]
                bond_feats += [bf, bf] # Bi-directional

            data = Data(
                x=x,
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float)
            )
            data_list.append(data)

        self._save_data(data_list, df, mean, std)

    @staticmethod
    def _get_bond_feature(bond):
        bt = bond.GetBondType()
        # Simple One-Hot Encoding
        return [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0
        ]


# --- VERSION 3 (Rich Atoms + Bond Attributes) ---
class SmallMoleculeFeaturizer_v3(SmallMoleculeFeaturizerBase):
    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v3.pt"] 

    def process(self):
        df, mean, std = self._load_and_clean_df()
        data_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V3"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None: continue

            # Rich Atom Features
            atom_feats = [self._get_atom_feature(atom) for atom in mol.GetAtoms()]
            x = torch.tensor(atom_feats, dtype=torch.float)

            # Edges WITH Attributes
            row_idx, col_idx, bond_feats = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = self._get_bond_feature(bond)
                
                row_idx += [start, end]
                col_idx += [end, start]
                bond_feats += [bf, bf]

            data = Data(
                x=x,
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float)
            )
            data_list.append(data)

        self._save_data(data_list, df, mean, std)

    @staticmethod
    def _get_bond_feature(bond):
        # Same as V2
        bt = bond.GetBondType()
        return [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0
        ]

    @staticmethod
    def _get_atom_feature(atom):
        # FIXED: Removed incompatible features and fixed Types
        return [
            float(atom.GetAtomicNum()),
            float(atom.GetMass() * 0.01),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetHybridization())), # Cast Enum to Int
            float(atom.GetIsAromatic()),         # 1.0 or 0.0
            float(atom.IsInRing()),              # 1.0 or 0.0
            float(atom.GetTotalNumHs())          # Useful feature you missed
        ]