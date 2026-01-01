import numpy as np
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data


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

        return edge_index.t()

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
