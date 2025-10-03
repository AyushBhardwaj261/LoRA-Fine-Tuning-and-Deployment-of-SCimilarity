import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import os

# --- THE FINAL CORRECT IMPORT ---
# The GeneEncoder class is directly in the top-level package.
from scimilarity import GeneEncoder

# --- Configuration ---
DATA_DIR = "data/"
MODEL_DIR = "models/"
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 5e-4

# --- Model Wrapper for Fine-Tuning (No changes needed) ---
class SCimilarityFinetuneModel(nn.Module):
    def __init__(self, scimilarity_model, num_classes):
        super().__init__()
        self.base_model = scimilarity_model
        embedding_dim = 512 # 'light' model embedding dimension
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embeds=False):
        embeds = self.base_model(x)
        outputs = self.classifier(embeds)
        
        if return_embeds:
            return (outputs, embeds)
        return outputs

# --- Dataset Definition (No changes needed) ---
class SingleCellDataset(Dataset):
    def __init__(self, adata, required_gene_order):
        self.adata = adata
        self.required_gene_order = required_gene_order
        self.y = torch.tensor(adata.obs['cell_type_id'].values, dtype=torch.long)
        self.gene_map = {gene: i for i, gene in enumerate(adata.var_names)}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        expression_vector = self.adata.X[idx].toarray().squeeze()
        aligned_vector = np.zeros(len(self.required_gene_order), dtype=np.float32)
        
        for i, gene in enumerate(self.required_gene_order):
            if gene in self.gene_map:
                aligned_vector[i] = expression_vector[self.gene_map[gene]]
                
        return torch.tensor(aligned_vector, dtype=torch.float32), self.y[idx]

def main():
    print("--- Starting Model Fine-Tuning with REAL SCimilarity Model (Final Corrected Import) ---")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # --- 1. Load Data and SCimilarity Model ---
    print("Loading preprocessed data...")
    adata_train = sc.read_h5ad(f"{DATA_DIR}train.h5ad")
    
    # --- Model loading (This part is correct and works with the right import) ---
    print("Loading SCimilarity pre-trained 'light' model...")
    scimilarity_base_model = GeneEncoder.from_pretrained("light")
    required_gene_order = scimilarity_base_model.gene_order

    # --- 2. Prepare Datasets ---
    train_dataset = SingleCellDataset(adata_train, required_gene_order)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(adata_train.obs['cell_type_id'].unique())
    print(f"Input dimensions (genes): {len(required_gene_order)}, Output classes: {num_classes}")

    # --- 3. Initialize Model, LoRA, and Freeze Weights ---
    print("Initializing model for fine-tuning...")
    base_model = SCimilarityFinetuneModel(scimilarity_base_model, num_classes)
    
    for param in base_model.base_model.parameters():
        param.requires_grad = False
        
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_fc", "c_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    print("Trainable parameters after applying LoRA:")
    model.print_trainable_parameters()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # --- 4. Training Loop ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = [b.to(device) for b in batch]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} average training loss: {avg_train_loss:.4f}")

    # --- 5. Save Model ---
    print("Saving LoRA adapters...")
    model.save_pretrained(f"{MODEL_DIR}lora_adapted_model")
    pd.Series(required_gene_order).to_json(f"{MODEL_DIR}gene_order.json")

    print("--- Training complete! ---")

if __name__ == "__main__":
    main()