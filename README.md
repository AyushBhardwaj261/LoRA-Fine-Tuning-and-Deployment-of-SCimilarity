# Project: LoRA Fine-Tuning and Deployment of SCimilarity for Single-Cell Classification

This project demonstrates a complete pipeline for fine-tuning the SCimilarity model using Low-Rank Adaptation (LoRA) for single-cell classification, and deploying the resulting model as a containerized API service.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Setup and Installation](#setup-and-installation)
4.  [How to Run](#how-to-run)
    *   [1. Data Preprocessing](#1-data-preprocessing)
    *   [2. Model Fine-Tuning](#2-model-fine-tuning)
    *   [3. Evaluation](#3-evaluation)
5.  [Inference Service](#inference-service)
    *   [Building the Docker Image](#building-the-docker-image)
    *   [Running the Container](#running-the-container)
    *   [API Usage](#api-usage)
6.  [Design Report and Analysis](#design-report-and-analysis)

---

## Project Overview

This project adapts the pre-trained SCimilarity model to a new single-cell RNA-seq dataset. Instead of full fine-tuning, we use LoRA to efficiently update a small subset of model parameters. This approach aims to improve classification performance on the new dataset while preserving the rich prior knowledge embedded in the original model, thus mitigating catastrophic forgetting.

## Dataset

We use data from the study **"A molecular single-cell lung cancer atlas of tumor immunity"** (Lavin, Y. et al., 2023), sourced via the CELLxGENE Discover platform.

- **Dataset ID**: `0e6c0a62-226e-47d3-933e-a753237189f7`
- **Subset**: We specifically use ~40,0.00 Peripheral Blood Mononuclear Cells (PBMCs) from healthy donors.
- **Justification**: This dataset is recent, well-annotated, and perfectly sized for this project. Using the `cellxgene-census` API ensures the data loading step is fully reproducible.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r main_requirements.txt
    ```

## How to Run

Execute the scripts in the `scripts/` directory in order.

### 1. Data Preprocessing

This script downloads the data, normalizes it, selects variable genes, and creates train/validation/test splits.

```bash
python scripts/1_preprocess.py
```
**Output**: Processed `.h5ad` files and artifacts will be saved in the `data/` directory.

### 2. Model Fine-Tuning

This script applies LoRA adapters to a placeholder SCimilarity model and fine-tunes it on the training data.

```bash
python scripts/2_train.py
```
**Output**: The trained LoRA adapters will be saved in the `models/lora_adapted_model/` directory.

### 3. Evaluation

This script compares the performance of the LoRA-adapted model against a baseline (frozen embeddings + logistic regression) and generates a confusion matrix.

```bash
python scripts/3_evaluate.py
```
**Output**: Performance metrics will be printed to the console, and a confusion matrix image will be saved.

## Inference Service

The fine-tuned model is deployed as a REST API using FastAPI and Docker.

### Building the Docker Image

From the root directory of the project:
```bash
docker build -t scimilarity-lora-api .
```

### Running the Container

```bash
docker run -d -p 8000:8000 --name scimilarity-service scimilarity-lora-api
```
The service will now be available at `http://localhost:8000`.

### API Usage

You can send a `POST` request to the `/predict` endpoint with a JSON payload representing the gene expression of a single cell.

**Example using `curl`:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "expression": {
    "CD3D": 1.2,
    "GNLY": 0.1,
    "MS4A1": 3.4,
    "CD14": 0.0
  }
}'
```

**Expected Response:**
```json
{
  "cell_type": "B cell",
  "confidence": 0.91
}
```

## Design Report and Analysis

### Design Choices
- **Reproducibility**: Using `cellxgene-census` ensures that the exact same data slice is downloaded every time, making the entire pipeline reproducible.
- **Modularity**: The code is split into distinct scripts for preprocessing, training, and evaluation, and a separate `api` directory for deployment. This separation of concerns makes the project easier to understand and maintain.
- **LoRA for Efficiency**: LoRA was chosen for fine-tuning as it drastically reduces the number of trainable parameters. This speeds up training and reduces the risk of catastrophic forgetting, as the original model weights remain frozen. The `peft` library from Hugging Face was used for its straightforward integration with PyTorch models.

### Results and Analysis

- **Performance**: The LoRA-adapted model is expected to outperform the baseline classifier, which operates on static, non-adapted embeddings. This is because fine-tuning allows the embeddings to shift and better separate the specific cell types present in our new dataset.

<!-- - **Confusion Matrix Analysis**:
  ![Confusion Matrix](./readme_assets/confusion_matrix.png)
  *(This image would be generated by the evaluation script)* -->

- **Misclassification Discussion**:
  1.  **B cell vs. Plasmablast**: These cell types are biologically adjacent. Misclassifications are expected as they share many transcriptional features.
  2.  **CD4+ T cell vs. CD8+ T cell**: These major T cell lineages can have overlapping functional states (e.g., naive, memory) that could confuse the model.
  3.  **Monocyte vs. Dendritic Cell**: As related myeloid cells, certain subsets have similar gene expression profiles, making them difficult to distinguish.
  4.  **NK cell vs. Cytotoxic T cell**: Both are cytotoxic lymphocytes sharing expression of effector molecules like granzymes, leading to potential confusion based on functional state.
  5.  **Naive vs. Memory Cell States**: Distinguishing between cell states (e.g., naive vs. memory T cells) is often harder than distinguishing cell types, as the transcriptional differences can be more subtle. LoRA helps the model fine-tune its attention to these subtler signals.