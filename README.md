# Skin Lesion Classification - ResNet-152

This project focuses on skin lesion classification using a ResNet-152 backbone.

#### Setup

1. Create and activate a virtual environment:
```bash
# Create
python -m venv .venv
# Activate on Windows
.venv\Scripts\activate
# Activate on Linux/MacOS
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Pipelines

**1. Data Preprocessing**
Organize and preprocess the raw HAM10000 dataset into the structured format required for DataLoader.
```bash
python -m src.pipelines.data_preprocessing
```

**2. Model Training**
Train the ResNet-152 model. The script automatically handles loading pretrained weights, applying class-weighted Cross-Entropy loss based on dataset imbalance, and batch-level minority augmentations depending on the `config.yaml` definitions.

To train the model in the development environment, run:
```bash
python -m src.pipelines.model_training --config config.dev.yaml
```
To train the model in the production environment, run:
```bash
python -m src.pipelines.model_training --config config.yaml
```

