from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold_split(
    dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42
):
    """
    Splits a PyG dataset by Bemis-Murcko scaffolds.
    """
    np.random.seed(seed)

    scaffold_to_indices = defaultdict(list)

    # 1. Group indices by Scaffold
    print("Generating Scaffolds...")
    for idx, data in enumerate(dataset):
        # We need the SMILES to generate the scaffold.
        # Ideally, store SMILES in the Data object or look it up in the original DF
        # Assuming dataset.df exists and aligns with dataset indices:
        smiles = dataset.df.iloc[idx]["SMILES"]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        # Get the core scaffold (removes side chains)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_to_indices[scaffold].append(idx)

    # 2. Sort scaffolds by size (to ensure balanced-ish splits)
    # We want to put rare scaffolds in train/val/test somewhat evenly,
    # but big clusters usually go to train to let the model learn.
    # A common strategy is simply randomizing the *groups*.

    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)

    train_indices = []
    val_indices = []
    test_indices = []

    train_cutoff = frac_train * len(dataset)
    val_cutoff = (frac_train + frac_val) * len(dataset)

    # 3. Fill the buckets
    current_count = 0

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]

        # If adding this whole cluster keeps us under the train cutoff, add to Train
        if len(train_indices) + len(indices) <= train_cutoff:
            train_indices.extend(indices)
        # Else if we are under the Val cutoff, add to Val
        elif len(train_indices) + len(val_indices) + len(indices) <= val_cutoff:
            val_indices.extend(indices)
        # Else, everything else goes to Test
        else:
            test_indices.extend(indices)

    print("Split Complete.")
    print(f"Train: {len(train_indices)} molecules")
    print(f"Val:   {len(val_indices)} molecules")
    print(f"Test:  {len(test_indices)} molecules")

    return train_indices, val_indices, test_indices
