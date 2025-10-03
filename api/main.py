from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import torch
import pandas as pd
import numpy as np
from peft import PeftModel

# Import the placeholder model from the training script's directory
# This requires adding the scripts directory to the path
import sys
sys.path.append('../scripts')
from train import SCimilarityPlaceholder

# --- App and Model Loading ---
app = FastAPI(title="SCimilarity LoRA Inference API")

# Configuration
ARTIFACTS_DIR = "../data/"
MODEL_DIR = "../models/"
DEVICE = torch.device("cpu") # Run on CPU for inference

# Load artifacts
try:
    GENE_LIST = pd.read_json(f"{ARTIFACTS_DIR}gene_list.json", typ='series').tolist()
    LABEL_MAPPING = pd.read_json(f"{ARTIFACTS_DIR}label_mapping.json", typ='series').to_dict()
    NUM_GENES = len(GENE_LIST)
    NUM_CLASSES = len(LABEL_MAPPING)
except FileNotFoundError:
    print("ERROR: Artifacts (gene_list.json, label_mapping.json) not found. Run preprocessing first.")
    sys.exit(1)

# Load the fine-tuned model
base_model = SCimilarityPlaceholder(input_dim=NUM_GENES, num_classes=NUM_CLASSES)
MODEL = PeftModel.from_pretrained(base_model, f"{MODEL_DIR}lora_adapted_model").to(DEVICE)
MODEL.eval()
print("Model loaded successfully.")

# --- API Data Models ---
class PredictionRequest(BaseModel):
    # Example: {"expression": {"CD3D": 1.2, "MS4A1": 3.4}}
    expression: Dict[str, float]

class PredictionResponse(BaseModel):
    cell_type: str
    confidence: float

# --- API Endpoints ---
@app.get("/", summary="API Root", description="Simple health check endpoint.")
def read_root():
    return {"status": "SCimilarity LoRA API is running"}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Cell Type")
def predict(request: PredictionRequest):
    """
    Accepts a cell-by-gene expression vector and returns the predicted cell type.
    
    The input should be a JSON object with an 'expression' key, which is a
    dictionary of gene_name: expression_value pairs.
    """
    # Create a zero vector with the correct gene order
    input_vector = pd.Series(0.0, index=GENE_LIST)
    
    # Fill the vector with the provided expression values
    # The .get(g, 0) ensures that if a gene is not in the request, it's treated as 0
    input_data = {g: request.expression.get(g, 0) for g in GENE_LIST}
    input_vector.update(pd.Series(input_data))

    # Convert to tensor for the model
    input_tensor = torch.tensor(input_vector.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = MODEL(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_label = LABEL_MAPPING.get(predicted_idx.item(), "Unknown")

    return PredictionResponse(
        cell_type=predicted_label,
        confidence=confidence.item()
    )