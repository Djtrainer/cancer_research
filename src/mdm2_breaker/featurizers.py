
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
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


class SmallMoleculeFeaturizer(InMemoryDataset):
    def __init__(self, root, file_path: str = None, transform=None, pre_transform=None):
        """
        Featurizes a small molecule into a PyG Data object.
        Args:
            root (str): Root directory of the dataset.
            file_path (str): Path to the file containing the small molecules.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before saving to disk.
        """
        self.file_path = file_path
        super(SmallMoleculeFeaturizer, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """
        PyG expects a value here, but we don't need it.
        """
        return []

    @property
    def processed_file_names(self):
        """
        Name of the processed file.
        """
        return ["mdm2_graphs.pt"]

    def download(self):
        """
        Download the dataset from the internet. (Not needed for this dataset)
        """
        pass

    def process(self):
        df = pd.read_csv(
            self.file_path, sep="\t", on_bad_lines="skip", low_memory=False
        )

        # Select important columns
        df = df[
            [
                "Ligand SMILES",
                "IC50 (nM)",
                "Target Name",
                "Target Source Organism According to Curator or DataSource",
            ]
        ]
        df.columns = ["SMILES", "IC50", "Target", "Organism"]

        # Filter for MDM2 + Human
        df = df[
            df["Target"].astype(str).str.contains("Mdm2", case=False)
            & df["Organism"].astype(str).str.contains("Homo sapiens", case=False)
        ]

        # Clean numeric IC50
        df["IC50_numeric"] = pd.to_numeric(
            df["IC50"].astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce"
        )

        # Calculate pIC50
        df["pIC50"] = -np.log10(df["IC50_numeric"] * 1e-9)

        # Drop junk
        df = df.dropna(subset=["pIC50", "SMILES"])

        global_mean = df["pIC50"].mean()
        global_std = df["pIC50"].std()
        
        df["pIC50_norm"] = (df["pIC50"] - global_mean) / global_std
        
        data_list = []
        allowed_atoms = [6, 7, 8, 9, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["SMILES"]
            pIC50_norm = float(row["pIC50_norm"])

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Filter for allowed atoms
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                if an in allowed_atoms:
                    node_feats.append(allowed_atoms.index(an))
                else:
                    node_feats.append(len(allowed_atoms))  # Other bucket

            x = torch.tensor(node_feats, dtype=torch.long).unsqueeze(1)

            # Edge Index (Bonds)
            row_idx, col_idx = [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # Add Bond A -> B
                row_idx.append(start)
                col_idx.append(end)
                # Add Bond B -> A 
                row_idx.append(end)
                col_idx.append(start)

            edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)

            # Create Data Object
            data = Data(
                x=x, edge_index=edge_index, y=torch.tensor([pIC50_norm]), dtype=torch.float
            )
            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No valid molecules found.")

        csv_path = self.processed_paths[0].replace('.pt', '.csv')
        df.to_csv(csv_path, index=False)
    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save({'mean': global_mean, 'std': global_std}, self.processed_paths[0].replace('.pt', '_stats.pt'))


class SmallMoleculeFeaturizer_v2(InMemoryDataset):
    def __init__(self, root, file_path: str = None, transform=None, pre_transform=None):
        """
        Featurizes a small molecule into a PyG Data object.
        Args:
            root (str): Root directory of the dataset.
            file_path (str): Path to the file containing the small molecules.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before saving to disk.
        """
        self.file_path = file_path
        super(SmallMoleculeFeaturizer_v2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """
        PyG expects a value here, but we don't need it.
        """
        return []

    @property
    def processed_file_names(self):
        """
        Name of the processed file.
        """
        return ["mdm2_graphs.pt"]

    def download(self):
        """
        Download the dataset from the internet. (Not needed for this dataset)
        """
        pass

    def process(self):
        df = pd.read_csv(
            self.file_path, sep="\t", on_bad_lines="skip", low_memory=False
        )

        # Select important columns
        df = df[
            [
                "Ligand SMILES",
                "IC50 (nM)",
                "Target Name",
                "Target Source Organism According to Curator or DataSource",
            ]
        ]
        df.columns = ["SMILES", "IC50", "Target", "Organism"]

        # Filter for MDM2 + Human
        df = df[
            df["Target"].astype(str).str.contains("Mdm2", case=False)
            & df["Organism"].astype(str).str.contains("Homo sapiens", case=False)
        ]

        # Clean numeric IC50
        df["IC50_numeric"] = pd.to_numeric(
            df["IC50"].astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce"
        )

        # Calculate pIC50
        df["pIC50"] = -np.log10(df["IC50_numeric"] * 1e-9)

        # Drop junk
        df = df.dropna(subset=["pIC50", "SMILES"])

        global_mean = df["pIC50"].mean()
        global_std = df["pIC50"].std()
        
        df["pIC50_norm"] = (df["pIC50"] - global_mean) / global_std
        
        def get_bond_feature(bond):
            # 0=Single, 1=Double, 2=Triple, 3=Aromatic
            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE: return [1,0,0,0]
            if bt == Chem.rdchem.BondType.DOUBLE: return [0,1,0,0]
            if bt == Chem.rdchem.BondType.TRIPLE: return [0,0,1,0]
            if bt == Chem.rdchem.BondType.AROMATIC: return [0,0,0,1]
            return [0,0,0,0]

        data_list = []
        allowed_atoms = [6, 7, 8, 9, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["SMILES"]
            pIC50_norm = float(row["pIC50_norm"])

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Filter for allowed atoms
            node_feats = []
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                if an in allowed_atoms:
                    node_feats.append(allowed_atoms.index(an))
                else:
                    node_feats.append(len(allowed_atoms))  # Other bucket

            x = torch.tensor(node_feats, dtype=torch.long).unsqueeze(1)

            # Edge Index (Bonds)
            row_idx, col_idx = [], []
            bond_features = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_features.append(get_bond_feature(bond))
                
                # Add Bond A -> B
                row_idx.append(start)
                col_idx.append(end)
                # Add Bond B -> A 
                row_idx.append(end)
                col_idx.append(start)

            edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)
            bond_features = torch.tensor(bond_features, dtype=torch.float)

            # Create Data Object
            data = Data(
                x=x, edge_index=edge_index, edge_attr=bond_features, y=torch.tensor([pIC50_norm]), dtype=torch.float
            )
            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No valid molecules found.")

        csv_path = self.processed_paths[0].replace('.pt', '.csv')
        df.to_csv(csv_path, index=False)
    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save({'mean': global_mean, 'std': global_std}, self.processed_paths[0].replace('.pt', '_stats.pt'))
