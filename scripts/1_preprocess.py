import scanpy as sc
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys

# --- Configuration ---
# IMPORTANT: UPDATE THESE TWO VARIABLES
# 1. Set the path to your local .h5ad file
LOCAL_H5AD_PATH = "CortexCultures4Days11DaysInVitro.h5ad"
# 2. Set the name of the column in adata.obs that contains cell type labels
CELL_TYPE_COLUMN = "cell_type" 

# --- Script Parameters ---
CELL_COUNT_DOWNSAMPLE = 40000
ARTIFACTS_DIR = "data/"
TEST_SPLIT_SIZE = 0.2

def main():
    """
    Main function to orchestrate data loading from a local file, 
    processing, and saving.
    """
    print("--- Starting Data Preprocessing from Local File ---")

    # --- 1. Load Local Data ---
    print(f"Loading local dataset from: {LOCAL_H5AD_PATH}")
    if not os.path.exists(LOCAL_H5AD_PATH):
        print(f"ERROR: File not found at '{LOCAL_H5AD_PATH}'. Please check the path.")
        sys.exit(1)
        
    adata = sc.read_h5ad(LOCAL_H5AD_PATH)

    # --- 2. Validate and Preprocess ---
    print(f"Original dataset size: {adata.n_obs} cells and {adata.n_vars} genes.")

    # Check if the specified cell type column exists
    if CELL_TYPE_COLUMN not in adata.obs.columns:
        print(f"ERROR: The column '{CELL_TYPE_COLUMN}' was not found in adata.obs.")
        print(f"Available columns are: {adata.obs.columns.tolist()}")
        sys.exit(1)

    # Ensure gene names are in the index (common practice)
    # If your gene names are in a column, you would set the index like this:
    # adata.var.index = adata.var['gene_symbols']
    adata.var_names_make_unique()


    # Subsample if the dataset is larger than the target count
    if adata.n_obs > CELL_COUNT_DOWNSAMPLE:
        print(f"Downsampling from {adata.n_obs} to {CELL_COUNT_DOWNSAMPLE} cells...")
        sc.pp.subsample(adata, n_obs=CELL_COUNT_DOWNSAMPLE, random_state=42)
        print(f"Downsampled to: {adata.n_obs} cells.")

    # Basic preprocessing
    print("Normalizing and log-transforming data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Selecting highly variable genes...")
    # Using seurat_v3 flavor is robust for pre-normalized, log-transformed data
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
    
    # Filter AnnData to keep only highly variable genes
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"Filtered to {adata.n_vars} highly variable genes.")

    # --- 3. Split Data ---
    print("Splitting data into train, validation, and test sets...")
    
    # Create integer labels for cell types for stratified splitting
    adata.obs['cell_type_id'] = adata.obs[CELL_TYPE_COLUMN].astype('category').cat.codes
    
    # Stratified split for train and a temporary set (val + test)
    train_indices, temp_indices = train_test_split(
        adata.obs.index,
        test_size=TEST_SPLIT_SIZE,
        stratify=adata.obs['cell_type_id'],
        random_state=42
    )
    
    # Split the temporary set equally into validation and test sets
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5, 
        stratify=adata.obs.loc[temp_indices, 'cell_type_id'],
        random_state=42
    )

    # Create AnnData objects for each split
    adata_train = adata[train_indices, :].copy()
    adata_val = adata[val_indices, :].copy()
    adata_test = adata[test_indices, :].copy()

    print(f"Train set size: {adata_train.n_obs}")
    print(f"Validation set size: {adata_val.n_obs}")
    print(f"Test set size: {adata_test.n_obs}")
    
    # --- 4. Save Artifacts ---
    print(f"Saving processed data artifacts to '{ARTIFACTS_DIR}'...")
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        
    adata_train.write(f"{ARTIFACTS_DIR}train.h5ad")
    adata_val.write(f"{ARTIFACTS_DIR}val.h5ad")
    adata_test.write(f"{ARTIFACTS_DIR}test.h5ad")
    
    # Save label mappings and gene list for the API
    # Use the specified cell type column
    label_mapping = dict(enumerate(adata.obs[CELL_TYPE_COLUMN].astype('category').cat.categories))
    pd.Series(label_mapping).to_json(f"{ARTIFACTS_DIR}label_mapping.json")
    
    gene_list = adata.var.index.tolist()
    pd.Series(gene_list).to_json(f"{ARTIFACTS_DIR}gene_list.json")

    print("--- Preprocessing complete! ---")

if __name__ == "__main__":
    main()