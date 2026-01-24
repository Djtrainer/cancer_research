import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# standard imports
import torch
from Bio.PDB import PDBParser
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


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
        df = pd.read_csv(
            self.file_path, sep="\t", on_bad_lines="skip", low_memory=False
        )

        # Select columns
        df = df[
            [
                "Ligand SMILES",
                "IC50 (nM)",
                "Target Name",
                "Target Source Organism According to Curator or DataSource",
            ]
        ]
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
            if mol is None:
                continue

            # V1 Node Features
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                node_feats.append(
                    allowed_atoms.index(an)
                    if an in allowed_atoms
                    else len(allowed_atoms)
                )

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
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float),
            )
            data_list.append(data)

        self._save_data(data_list, df, mean, std)


# --- VERSION 2 (Simple Atoms + Bond Attributes) ---
class SmallMoleculeFeaturizer_v2(SmallMoleculeFeaturizerBase):
    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v2.pt"]  #

    def process(self):
        df, mean, std = self._load_and_clean_df()
        data_list = []
        allowed_atoms = [6, 7, 8, 9, 16, 17, 35, 53]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V2"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue

            # Nodes (Same as V1)
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                node_feats.append(
                    allowed_atoms.index(an)
                    if an in allowed_atoms
                    else len(allowed_atoms)
                )
            x = torch.tensor(node_feats, dtype=torch.long).unsqueeze(1)

            # Edges WITH Attributes
            row_idx, col_idx, bond_feats = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = self._get_bond_feature(bond)

                row_idx += [start, end]
                col_idx += [end, start]
                bond_feats += [bf, bf]  # Bi-directional

            data = Data(
                x=x,
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float),
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
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]


# --- VERSION 3 (Rich Atoms + Bond Attributes) ---
class SmallMoleculeFeaturizer_v3(SmallMoleculeFeaturizerBase):
    def __init__(self, root, file_path, test=False, transform=None, pre_transform=None):
        self.file_path = file_path
        self.test = test

        super().__init__(root, file_path, transform, pre_transform)

        # Load the DF into memory permanently
        self.df = self._load_df()

    def _load_df(self):
        """Helper to load the cleaned CSV if available, else raw TSV."""
        # This matches the filename we save in process() below
        processed_csv = self.processed_paths[0].replace(".pt", ".csv")

        if os.path.exists(processed_csv):
            return pd.read_csv(processed_csv)
        else:
            return pd.read_csv(self.file_path, sep="\t")

    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v3.pt"]

    def process(self):
        # 1. Load shared cleaning logic
        df, mean, std = self._load_and_clean_df()
        data_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V3"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue

            # --- 1. NODE FEATURES (Updated) ---
            atom_features_list = []
            for atom in mol.GetAtoms():
                # Get the tuple: ([cats], [floats])
                cats, floats = self._get_atom_feature(atom)

                # Flatten into a single list: [z, hyb, chir, mass, deg, charge...]
                # We perform this concatenation so we can store it in a single tensor 'x'
                combined_features = cats + floats
                atom_features_list.append(combined_features)

            # Create Tensor. Note: Everything must be float for now.
            # The model will cast the first 3 columns back to .long() later.
            x = torch.tensor(atom_features_list, dtype=torch.float)

            # --- 2. EDGE FEATURES (Standard) ---
            row_idx, col_idx, bond_feats = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = self._get_bond_feature(bond)

                # Add Bi-directional edges
                row_idx += [start, end]
                col_idx += [end, start]
                bond_feats += [bf, bf]

            # --- 3. CREATE DATA OBJECT ---
            data = Data(
                x=x,
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float),
            )
            data_list.append(data)

        # 4. Save using the shared base helper
        self._save_data(data_list, df, mean, std)

    @staticmethod
    def _get_atom_feature(atom):
        # --- CATEGORICAL (Indices for Embeddings) ---
        # 1. Atomic Num (0-100)
        z = atom.GetAtomicNum()

        # 2. Hybridization (Map to 0, 1, 2, 3...)
        hyb_map = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4,
        }
        hyb = hyb_map.get(atom.GetHybridization(), 5)  # 5 = Other

        # 3. Chirality (0, 1, 2)
        chirality = 0
        if atom.HasProp("_CIPCode"):
            code = atom.GetProp("_CIPCode")
            if code == "R":
                chirality = 1
            elif code == "S":
                chirality = 2

        # --- CONTINUOUS (Floats) ---
        mass = atom.GetMass() * 0.01
        degree = atom.GetDegree() / 10.0
        charge = atom.GetFormalCharge()  # Keep raw, linear layer handles sign
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        is_in_ring = 1.0 if atom.IsInRing() else 0.0
        num_hs = atom.GetTotalNumHs() / 5.0

        # Return Tuple: (Categorical List, Continuous List)
        return [z, hyb, chirality], [
            mass,
            degree,
            charge,
            is_aromatic,
            is_in_ring,
            num_hs,
        ]

    @staticmethod
    def _get_bond_feature(bond):
        # Same as V2
        bt = bond.GetBondType()
        return [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]


# --- VERSION 5 (Rich Atoms + Bond Attributes + Fingerprints Model) ---
class SmallMoleculeFeaturizer_v5(SmallMoleculeFeaturizer_v3):
    @property
    def processed_file_names(self):
        return ["mdm2_graphs_v5.pt"]

    def process(self):
        # 1. Load & Clean (Shared logic from Base)
        df, mean, std = self._load_and_clean_df()

        data_list = []

        # Generator for Morgan Fingerprints (2048 bits, Radius 2)
        # This is faster than calling AllChem.GetMorganFingerprintAsBitVect repeatedly
        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing V4 (Hybrid)"):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue

            # --- A. GRAPH FEATURES (V3 Logic) ---
            # 1. Nodes: Combine Categorical + Continuous features
            atom_feats = []
            for atom in mol.GetAtoms():
                cats, floats = self._get_atom_feature(atom)
                atom_feats.append(cats + floats)
            x = torch.tensor(atom_feats, dtype=torch.float)

            # 2. Edges: Bond Types
            row_idx, col_idx, bond_feats = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = self._get_bond_feature(bond)

                # Bi-directional
                row_idx += [start, end]
                col_idx += [end, start]
                bond_feats += [bf, bf]

            # --- B. EXPERT FEATURES (The New Hybrid Part) ---
            # 1. Morgan Fingerprint (2048 bits)
            # Returns a numpy array of 0s and 1s
            fp = fp_gen.GetFingerprintAsNumPy(mol)

            # 2. Global Descriptors (4 floats)
            # We scale them roughly to 0-1 range to help the neural network
            descriptors = np.array(
                [
                    Descriptors.MolWt(mol) * 0.001,  # e.g., 500 -> 0.5
                    Descriptors.MolLogP(mol) * 0.1,  # e.g., 3.0 -> 0.3
                    Descriptors.TPSA(mol) * 0.01,  # e.g., 100 -> 1.0
                    Descriptors.NumHDonors(mol) * 0.1,  # e.g., 2 -> 0.2
                ],
                dtype=np.float32,
            )

            # Concatenate: [2048] + [4] = [2052]
            expert_vec = np.concatenate([fp, descriptors])

            # --- C. PACK EVERYTHING ---
            data = Data(
                x=x,
                edge_index=torch.tensor([row_idx, col_idx], dtype=torch.long),
                edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                y=torch.tensor([row["pIC50"]], dtype=torch.float),
                # IMPORTANT: Store as [1, 2052] so it has a batch dimension
                expert_features=torch.tensor([expert_vec], dtype=torch.float),
            )
            data_list.append(data)

        # 3. Save to disk
        self._save_data(data_list, df, mean, std)

    def _featurize_mol(self, mol):
        """
        Core logic: Takes RDKit Mol -> Returns PyG Data components
        """
        # --- A. GRAPH FEATURES (Nodes & Edges) ---
        atom_feats = []
        for atom in mol.GetAtoms():
            cats, floats = self._get_atom_feature(atom)
            atom_feats.append(cats + floats)
        x = torch.tensor(atom_feats, dtype=torch.float)

        row_idx, col_idx, bond_feats = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = self._get_bond_feature(bond)
            row_idx += [start, end]
            col_idx += [end, start]
            bond_feats += [bf, bf]
        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        # --- B. EXPERT FEATURES ---
        fp = fp_gen.GetFingerprintAsNumPy(mol)

        # EXACT MATCH of your V5 Logic (Don't change these constants!)
        descriptors = np.array(
            [
                Descriptors.MolWt(mol) * 0.001,
                Descriptors.MolLogP(mol) * 0.1,
                Descriptors.TPSA(mol) * 0.01,
                Descriptors.NumHDonors(mol) * 0.1,
            ],
            dtype=np.float32,
        )

        expert_vec = np.concatenate([fp, descriptors])

        return x, row_idx, col_idx, bond_feats, expert_vec

    def featurize_smiles(self, smiles):
        """
        Public method for Design/Inference scripts.
        """
        mol = Chem.MolFromSmiles(smiles)
        # Note: Your training data might have explicit hydrogens.
        # If your AtomEncoder relies on H-counts, AddHs is safer.
        # But if your V5 training loop didn't use AddHs(), don't use it here!
        # Assuming standard V5 didn't force AddHs:

        if mol is None:
            return None

        x, row, col, bond_attr, expert = self._featurize_mol(mol)

        # Wrap in Data object (No y label needed for inference)
        data = Data(
            x=x,
            edge_index=torch.tensor([row, col], dtype=torch.long),
            edge_attr=torch.tensor(bond_attr, dtype=torch.float),
            expert_features=torch.tensor([expert], dtype=torch.float),  # [1, 2052]
        )
        return data


class SmallMoleculeFeaturizer_DeepPurpose(SmallMoleculeFeaturizer_v3):
    """
    Official DGL Implementation.
    - Uses dgllife.utils.CanonicalAtomFeaturizer for exact 74-dim features.
    - Matches DeepPurpose 'DGL_GCN' encoding.
    """

    def __init__(self, root, file_path, test=False, transform=None, pre_transform=None):
        # Initialize the DGL Featurizer
        # atom_data_field='h' tells it to store features in the 'h' key
        self.backend = CanonicalAtomFeaturizer(atom_data_field="h")

        # Initialize Parent
        super().__init__(root, file_path, test, transform, pre_transform)

    @property
    def processed_file_names(self):
        # We name it _dgl to distinguish from the RDKit workaround version
        return ["mdm2_graphs_dgl.pt"]

    def process(self):
        # 1. Load & Clean (Shared logic)
        df, mean, std = self._load_and_clean_df()
        data_list = []

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing DeepPurpose (DGL)"
        ):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue

            mol = self._get_largest_frag(mol)
            # --- A. GRAPH FEATURES VIA DGL ---
            try:
                # 1. Nodes: Use DGL Backend
                # returns a dict: {'h': tensor_of_shape_N_74}
                feats_dict = self.backend(mol)
                x = feats_dict["h"].float()

                # 2. Edges: Connectivity Only (Standard RDKit Adjacency)
                # DeepPurpose GCN does not use edge features
                adj = Chem.GetAdjacencyMatrix(mol)
                edge_index = torch.tensor(adj).nonzero(as_tuple=False).t().contiguous()

            except Exception as e:
                print(f"DGL Featurization failed for {row['SMILES']}: {e}")
                continue

            # --- B. PACK ---
            data = Data(
                x=x,
                edge_index=edge_index.long(),
                y=torch.tensor([row["pIC50_norm"]], dtype=torch.float),
            )
            data_list.append(data)

        self._save_data(data_list, df, mean, std)

    def featurize_smiles(self, smiles):
        """
        Public method for Design/Inference.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Nodes via DGL
        feats_dict = self.backend(mol)
        x = feats_dict["h"].float()

        # 2. Edges
        adj = Chem.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(adj).nonzero(as_tuple=False).t().contiguous()

        # Wrap
        data = Data(
            x=x,
            edge_index=edge_index.long(),
        )
        return data

    @staticmethod
    def _get_largest_frag(mol):
        """
        Removes salts/solvents by keeping only the largest organic fragment.
        """
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            # Return the fragment with the most atoms
            return max(frags, key=lambda m: m.GetNumAtoms())
        return mol

