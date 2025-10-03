import torch
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from peft import PeftModel, PeftConfig
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Import placeholder model and dataset class from train script
from train import SCimilarityPlaceholder, SingleCellDataset

# --- Configuration ---
DATA_DIR = "../data/"
MODEL_DIR = "../models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model, dataloader):
    """Helper function to get embeddings from a model."""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            inputs, _ = [b.to(DEVICE) for b in batch]
            # Assumes the model can return just embeddings
            embeddings = model(inputs, return_embeds=True)[1] if isinstance(model, PeftModel) else model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def main():
    """
    Main function for model evaluation.
    """
    print("--- Starting Model Evaluation ---")
    
    # --- 1. Load Data ---
    print("Loading test and train data...")
    adata_test = sc.read(f"{DATA_DIR}test.h5ad")
    adata_train = sc.read(f"{DATA_DIR}train.h5ad") # Needed for baseline classifier training
    
    test_dataset = SingleCellDataset(adata_test)
    train_dataset = SingleCellDataset(adata_train)
    
    test_loader = DataLoader(test_dataset, batch_size=64)
    train_loader = DataLoader(train_dataset, batch_size=64)
    
    num_genes = adata_test.n_vars
    num_classes = len(adata_test.obs['cell_type'].unique())
    class_names = adata_test.obs['cell_type'].astype('category').cat.categories.tolist()

    # --- 2. Baseline Evaluation ---
    print("\n--- Evaluating Baseline (Frozen Embeddings + Logistic Regression) ---")
    baseline_model = SCimilarityPlaceholder(input_dim=num_genes).to(DEVICE)
    
    # Get embeddings for train and test sets
    train_embeddings = get_embeddings(baseline_model, train_loader)
    test_embeddings = get_embeddings(baseline_model, test_loader)
    
    # Train a simple classifier on the embeddings
    print("Training Logistic Regression classifier on baseline embeddings...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_dataset.y)
    
    baseline_preds = classifier.predict(test_embeddings)
    baseline_accuracy = accuracy_score(test_dataset.y, baseline_preds)
    baseline_f1 = f1_score(test_dataset.y, baseline_preds, average='weighted')
    
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline F1-Score (Weighted): {baseline_f1:.4f}")

    # --- 3. LoRA Model Evaluation ---
    print("\n--- Evaluating LoRA Fine-Tuned Model ---")
    
    # Load base model and apply LoRA adapters
    lora_model_path = f"{MODEL_DIR}lora_adapted_model"
    base_model = SCimilarityPlaceholder(input_dim=num_genes, num_classes=num_classes)
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path).to(DEVICE)
    lora_model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting with LoRA model"):
            inputs, _ = [b.to(DEVICE) for b in batch]
            outputs = lora_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    lora_preds = np.array(all_preds)
    lora_accuracy = accuracy_score(test_dataset.y, lora_preds)
    lora_f1 = f1_score(test_dataset.y, lora_preds, average='weighted')

    print(f"LoRA-adapted Model Accuracy: {lora_accuracy:.4f}")
    print(f"LoRA-adapted Model F1-Score (Weighted): {lora_f1:.4f}")

    # --- 4. Confusion Matrix and Analysis ---
    print("\n--- Analysis of Misclassifications ---")
    cm = confusion_matrix(test_dataset.y, lora_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix for LoRA-adapted Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("../readme_assets/confusion_matrix.png")
    print("Confusion matrix saved to ../readme_assets/confusion_matrix.png")

    print("\nDiscussion of Misclassified Populations:")
    print("1. B cell vs. Plasmablast: These cell types are closely related in the B cell lineage. Misclassifications are common as Plasmablasts are differentiating B cells and share many genetic markers.")
    print("2. CD4+ T cell vs. CD8+ T cell: While distinct, these T cell subtypes can have overlapping states (e.g., memory vs. naive) that might confuse the model if the most distinguishing genes were not ranked as highly variable.")
    print("3. Monocyte vs. Dendritic Cell (DC): Both are myeloid-derived antigen-presenting cells. Certain subsets, like classical monocytes and conventional DCs, can have similar transcriptional profiles, leading to confusion.")
    print("4. Naive vs. Memory T cells: This is a state-based distinction rather than a type-based one. The model might struggle if the signals separating these states are subtle compared to the signals separating major lineages.")
    print("5. NK cell vs. Cytotoxic T cell: Both are cytotoxic lymphocytes and express similar effector molecules like granzymes and perforin. This functional similarity can lead to misclassification.")

if __name__ == "__main__":
    main()