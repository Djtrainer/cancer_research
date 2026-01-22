from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import py3Dmol
import seaborn as sns
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
from torch_geometric.utils import to_networkx


def view_protein(pdb_file, highlight_chain="A"):
    view = py3Dmol.view(query=pdb_file)

    # Show the whole protein as a "Cartoon" (Ribbon)
    view.setStyle({"cartoon": {"color": "spectrum"}})

    # Show the Alpha Carbons as spheres
    view.addStyle(
        {"chain": highlight_chain, "atom": "CA"},
        {"sphere": {"radius": 0.5, "color": "red"}},
    )

    view.zoomTo()
    view.show()


def visualize_graph(data, title="Molecule Graph"):
    # 1. Convert PyG Data to NetworkX
    # to_undirected ensures we don't draw two arrows for every bond
    G = to_networkx(data, to_undirected=True)

    # 2. Map the "Node Features" back to Element Names
    # Your allowed_atoms list from the featurizer:
    atom_map = {
        0: "C",
        1: "N",
        2: "O",
        3: "F",
        4: "S",
        5: "Cl",
        6: "Br",
        7: "I",
        8: "?",
    }

    # Create labels and colors for the plot
    labels = {}
    node_colors = []

    for node_idx in G.nodes():
        # Get the feature index from the tensor
        # data.x[node_idx] is a tensor like [0], we need the integer 0
        feat_idx = data.x[node_idx].item()

        element = atom_map.get(feat_idx, "?")
        labels[node_idx] = f"{node_idx}:{element}"  # Format: "Index:Element"

        # Color logic (CPK-ish coloring)
        if element == "C":
            node_colors.append("gray")
        elif element == "N":
            node_colors.append("blue")
        elif element == "O":
            node_colors.append("red")
        elif element == "S":
            node_colors.append("yellow")
        elif element == "Cl":
            node_colors.append("lightgreen")
        else:
            node_colors.append("orange")

    # 3. Draw it
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # Force-directed layout

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=800,
        font_color="white",
        font_weight="bold",
        edge_color="black",
    )

    plt.title(f"{title}\npIC50: {data.y.item():.2f}")
    plt.show()


def visualize_scaffold_split(dataset, train_idx, val_idx, test_idx):
    """
    Generates a 3-part visualization of the scaffold split.
    1. Grid of Top Train vs. Top Test Scaffolds (Structural Distinctness)
    2. t-SNE plot of Chemical Space (Distribution)
    3. Distribution of pIC50s (Label Shift)
    """

    # --- Helper: Get Scaffold & Fingerprint ---
    def get_info(indices, label):
        scaffolds = []
        fps = []
        pics = []

        fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        for idx in indices:
            smiles = dataset.df.iloc[idx]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Get Scaffold
                scaff = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
                scaffolds.append(scaff)

                # Get Fingerprint (for t-SNE)
                fp = fp_gen.GetFingerprintAsNumPy(mol)
                fps.append(fp)

                # Get Label
                pics.append(dataset.df.iloc[idx]["pIC50"])

        return scaffolds, np.array(fps), np.array(pics)

    print("Analyzing Train Set...")
    train_scaffs, train_fps, train_y = get_info(train_idx, "Train")
    print("Analyzing Test Set...")
    test_scaffs, test_fps, test_y = get_info(test_idx, "Test")

    # --- PLOT 1: The "Face" of the Split (Top Scaffolds) ---
    def draw_top_scaffolds(scaff_list, title, n=5):
        counts = Counter(scaff_list)
        top_scaffs = counts.most_common(n)

        mols = [Chem.MolFromSmiles(s) for s, c in top_scaffs]
        legends = [f"Count: {c}" for s, c in top_scaffs]

        img = Draw.MolsToGridImage(
            mols, molsPerRow=n, subImgSize=(200, 200), legends=legends
        )
        return img

    print(f"\n--- Top {5} Training Scaffolds ---")
    display(draw_top_scaffolds(train_scaffs, "Top Train Scaffolds"))

    print(f"\n--- Top {5} Test Scaffolds (Should be different!) ---")
    display(draw_top_scaffolds(test_scaffs, "Top Test Scaffolds"))

    # --- PLOT 2: Chemical Space t-SNE ---
    print("\nComputing t-SNE (this might take a moment)...")

    # Combine for t-SNE to ensure shared space
    X_all = np.vstack([train_fps, test_fps])
    labels = ["Train"] * len(train_fps) + ["Test"] * len(test_fps)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    X_embedded = tsne.fit_transform(X_all)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels,
        style=labels,
        alpha=0.6,
        palette={"Train": "blue", "Test": "red"},
    )
    plt.title("Chemical Space of Scaffold Split")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

    # --- PLOT 3: Label Distribution ---
    plt.figure(figsize=(8, 5))
    sns.kdeplot(train_y, fill=True, label="Train", color="blue", alpha=0.3)
    sns.kdeplot(test_y, fill=True, label="Test", color="red", alpha=0.3)
    plt.title("pIC50 Distribution (Check for Distribution Shift)")
    plt.xlabel("pIC50")
    plt.legend()
    plt.show()
