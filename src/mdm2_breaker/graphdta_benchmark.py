import numpy as np
import pandas as pd
from DeepPurpose import utils, DTI
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
import matplotlib.pyplot as plt

mol_file = os.path.join(
    "data", "MDM2_Breaker", "raw", "bindingdb_p53_binding_protein_mdm2.tsv"
)
print(f"mol_file: {mol_file}")
BENCHMARK_DATA_PATH = os.path.join('data', 'MDM2_Breaker', 'processed', 'benchmark_data.csv')
print(f"BENCHMARK_DATA_PATH: {BENCHMARK_DATA_PATH}")
df_mol = pd.read_csv(BENCHMARK_DATA_PATH)
print(f"df_mol: {df_mol.head()}")

print(f"Original data size: {len(df_mol)}")

def is_valid_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# Drop rows with invalid SMILES
df_mol = df_mol[df_mol['SMILES'].apply(is_valid_molecule)]

def sanitize_and_flatten(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # CRITICAL FIX: Remove the "Wedge/Dash" bond directions
        # This prevents the 'BEGINDASH' error in DGL-LifeSci
        Chem.RemoveStereochemistry(mol)
        
        # Return the canonical, flattened SMILES
        return Chem.MolToSmiles(mol)
    except:
        return None

# 1. Apply the cleaner
df_mol['SMILES'] = df_mol['SMILES'].apply(sanitize_and_flatten)
# 2. Drop rows that failed (became None)
df_mol = df_mol.dropna(subset=['SMILES'])
print(f"Sanitized & Flattened data size: {len(df_mol)}")


train_indices = df_mol[df_mol['split'] == 'train'].index
val_indices = df_mol[df_mol['split'] == 'val'].index
test_indices = df_mol[df_mol['split'] == 'test'].index
print(f"train_indices: {train_indices}")
print(f"val_indices: {val_indices}")
print(f"test_indices: {test_indices}")
# Define the MDM2 Sequence (Standardized)
MDM2_SEQUENCE = "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN"

# 2. Use this function to generate the 3 lists required
def prepare_data_for_graphdta(df, indices, target_seq):
    subset = df.loc[indices]
    
    X_drugs = subset['SMILES'].tolist()           # 1. Drug (SMILES)
    y = subset['pIC50_norm'].tolist()                  # 2. Label (pIC50)
    X_targets = [target_seq] * len(X_drugs)       # 3. Target (Repeated Sequence)
    
    return X_drugs, X_targets, y

# 3. Call it for each split
print("Preparing Data...")
X_train, T_train, y_train = prepare_data_for_graphdta(df_mol, train_indices, MDM2_SEQUENCE)
X_val, T_val, y_val = prepare_data_for_graphdta(df_mol, val_indices, MDM2_SEQUENCE)
X_test, T_test, y_test = prepare_data_for_graphdta(df_mol, test_indices, MDM2_SEQUENCE)

print(f"Data ready: {len(X_train)} Train, {len(X_val)} Val, {len(X_test)} Test")

# 3. Encode (The Heavy Lifting)
drug_encoding = 'DGL_GCN' # The standard GraphDTA graph encoder
# drug_encoding = 'DGL_GIN_AttrMasking' # The standard GraphDTA graph encoder
target_encoding = 'CNN'   # The standard 1D protein encoder

train_data = utils.data_process(X_drug=X_train, X_target=T_train, y=y_train, 
                                drug_encoding=drug_encoding, target_encoding=target_encoding, 
                                split_method='no_split')

val_data = utils.data_process(X_drug=X_val, X_target=T_val, y=y_val, 
                              drug_encoding=drug_encoding, target_encoding=target_encoding, 
                              split_method='no_split')

test_data = utils.data_process(X_drug=X_test, X_target=T_test, y=y_test, 
                               drug_encoding=drug_encoding, target_encoding=target_encoding, 
                               split_method='no_split')

# 4. Train
config = utils.generate_config(drug_encoding=drug_encoding, 
                               target_encoding=target_encoding, 
                               cls_hidden_dims=[1024, 1024, 512], 
                               train_epoch=30, # 30 is usually enough for convergence
                               LR=0.001, 
                               batch_size=128)

model = DTI.model_initialize(**config)


# --- ARCHITECTURE INSPECTION ---
network = model.model 

print("\n=== DEEPPURPOSE MODEL ARCHITECTURE ===")
print(network) 

# 2. Count parameters on the NETWORK, not the wrapper
total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"\nTotal Trainable Parameters: {total_params:,}")
print("======================================\n")
print("Starting Training...")
model.train(train_data, val_data, test_data)

# Define a directory for the saved model
save_path = os.path.join('models', 'MDM2_Breaker',  'DeepPurpose_Benchmark')

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Save the model architecture and weights
model.save_model(save_path)
print(f"\nModel successfully saved to: {save_path}")

# --- NEW: SAVE LOSS PLOT ---
plt.figure(figsize=(10, 6))
plt.plot(model.train_losses, label='Training Loss', color='blue')
plt.plot(model.val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot to the same directory as the model
plot_path = os.path.join(save_path, 'loss_curve.png')
plt.savefig(plot_path, dpi=300)
plt.close() # Close figure to free memory

print(f"Loss plot saved to: {plot_path}")
# ---------------------------

# 5. Get The Final Number
print("\n--- Final Test Performance ---")
y_pred = model.predict(test_data)
df_results = pd.DataFrame({'SMILES': X_test, 'pIC50': y_test, 'pIC50_pred': y_pred})
df_results.to_csv(os.path.join('data', 'MDM2_Breaker', 'processed', 'graphdta_benchmark_results_gin_attr_masking.csv'), index=False)

# DeepPurpose prints MSE/Pearson automatically, but let's be sure
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")