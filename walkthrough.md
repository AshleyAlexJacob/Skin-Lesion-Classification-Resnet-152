# Walkthrough - Dataset Splitting and Data Loaders Updates

## Goal 
The objective was to implement a script to create a physical Train/Val/Test (63/27/10 ratio) split of the dataset and update the respective model pipelines to rely on these generated directories instead of random runtime dataset splitting. 

## Changes Made

### 1. Data Split Script (`src/data/split_dataset.py`)
- Created a `split_data(input_dir, output_dir, train_ratio, val_ratio, test_ratio, seed)` function to safely distribute images from class subdirectories into a stratified structure.

### 2. Splitting Pipeline Script (`src/pipelines/split_pipeline.py`)
- Following project rules, the modification of `src/pipelines/data_preprocessing.py` was reverted, and a separate `split_pipeline.py` script was created to invoke `split_data`.
- This script is completely separate from standard preprocessing. 
- You can execute it via: `python -m src.pipelines.split_pipeline`

### 3. Pipeline Data Loaders Updates
- **`src/data/loader.py`**: Changed the `get_data_loaders` signature to accept explicit `train_dir` and `val_dir` paths. This eliminates the uncertainty of PyTorch's `random_split` since datasets are loaded via `datasets.ImageFolder` using pre-assigned images.
- **`src/pipelines/model_training.py`**: Updated the trainer's `__init__` constructor to derive the `train` and `val` paths from the base `data_dir` configuration mapping. These explicit mappings are passed down into `get_data_loaders`.
- **`src/model/evaluation.py`**: Removed the complicated fallback mechanism querying the legacy training datasets since checking the specific `test` sub-directory directly achieves the goal natively.

## What Was Tested
- **Documentation (`README.md`)**: Replaced dataset processing instructions to highlight the new `src/pipelines/split_pipeline.py` execution.

## Process overview: How Changes are Made
In this project, we adhere to strict development rules as defined in the `CLAUDE.md` file:
1. **Single Responsibility & Non-Destructive Modifying**: Instead of changing the original core `data_preprocessing.py`, we created an isolated script (`split_pipeline.py`) dedicated to running the splits to maintain modularity.
2. **Explicit User Permissions**: Any change to pre-existing files (e.g., updating the loaders inside `src/data/loader.py` and `src/pipelines/model_training.py`) must be explicitly approved by the user via an `implementation_plan.md` review beforehand. 
3. **Strict Formatting & Linting**: All implemented code must comply with PEP8, enforced via `ruff check` and `ruff format`. Type hinting ensures type safety.
4. **Reproducibility**: Random generator seeds (like `seed=42`) were utilized to ensure uniform splits every single time the pipeline is executed.

## Usage of Split Data
- **Train & Val splits**: Used exclusively during the training phase (`src/pipelines/model_training.py`) to learn patterns and tune the model's performance over epochs.
- **Test split**: Used exclusively for model evaluation (`src/pipelines/model_evaluation.py` via `src/model/evaluation.py`). It is held completely separate from training and is only evaluated at the very end to judge the model's generalization on completely unseen data. These final evaluation results are saved automatically to `artifacts/results/testing/`.

---

# Walkthrough - FastAPI Prediction Service Implementation

The original CLI/Streamlit-based scripts have been supplemented with a fully structured and robust FastAPI predicting service for the ResNet-152 model.

## File Architecture (`app/` Directory Breakdown)

The service is modularly designed, separating different architectural concerns into specific files within the `app/` directory:

- **`app/schemas.py`**: Defines explicit Pydantic data models (e.g., `PredictionResponse`). This handles data validation and auto-generates the input/output schemas for the OpenAPI (Swagger) documentation.
- **`app/model_loader.py`**: Encapsulates the PyTorch model instantiation and weight loading. It constructs the ResNet-152 architecture with our 7 skin lesion classes and reliably loads the pre-trained weights from `best_model.pth`.
- **`app/dependencies.py`**: Defines reusable FastAPI dependency generators. It strategically uses `@lru_cache` to load the heavy ResNet model and image transforms into memory exactly once at application startup. This provides efficient and fast dependency injection for the API routes.
- **`app/predict.py`**: Isolates the core inference business logic from the API endpoints. It processes raw image bytes, applies necessary `torchvision` transforms, runs the forward pass, and calculates Softmax probabilities to return the highest confidence prediction.
- **`app/middleware.py`**: Centralizes application-level logic outside of explicit endpoints. It implements Cross-Origin Resource Sharing (CORS), sets up a global exception handler for graceful JSON error reporting, and adds custom request logging for timing each API call.
- **`app/main.py`**: The central application factory and entry point. It creates the FastAPI instance using `create_app()`, applies the middleware, handles the dependency injection, and wires up the routing (`/` and `/predict`), rendering a simple HTML UI alongside the predicting endpoint.

## Implementation Validation
- Executed `ruff check app/ --fix && ruff format app/` fixing auto-formatting quirks.
- Conducted an isolated runtime test confirming that `.venv` can load the full `fastapi` machinery alongside our `torch` model logic accurately.
- Required dependencies like `python-multipart` (for file uploads) and `fastapi` were added via `uv`.

## Dark Mode Medical Dashboard
- Upgraded the static HTML endpoint `/` to a **Premium, Widget-based Dark Mode Medical Dashboard** utilizing React, TailwindCSS, and Phosphor Icons directly via CDN.
- **Architectural Layout:**
  - **Fluid Bento-Grid Layout:** Responsive left-to-right flow with dedicated modular cards featuring glassy background blur and subtle blue atmospheric glows.
  - **Dynamic Action Hero:** A prominent *Surgical Blue* gradient card handles drag-and-drop image uploads and triggers the analysis pipeline.
- **Advanced Micro-Interactions:**
  - **Hover Telemetry:** Interactive `translate-y` POP states on all cards with dynamic teal casting shadows.
  - **Data Visualization:** Animated laser-scanning overlay during image processing and an animated, custom-drawn SVG EKG heartbeat waveform bridging empty data states.
  - **Pulsing States:** Simulated medical loading via staggered skeleton placeholders, fading in data rows cleanly on completion, alongside a smooth semi-circular SVG confidence gauge.
- **Clinical Metrics:** Cleanly splits raw `result.prediction` data into detailed clinical breakdowns (Risk Severity, Medical Definitions, Save/Review actions).

## Getting Started
You can easily spin up the environment straight from `uvicorn`:
```bash
# Start the server:
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
You can access the built-in Interactive Medical UI at `http://localhost:8000/` or visit Swagger at `http://localhost:8000/docs`.

---

# Walkthrough - UI Restoration

After the advanced animation enhancements in a prior session, certain UI elements were reverted at the user's request to restore a preferred previous state.

## Changes Reverted in `app/templates/index.html`

- **Header Icons**: Restored the original Bell, Squares-four, and Gear icons in the top-right header.
- **Image Action Buttons**: Reverted the four icon buttons below the upload area back to Folder, Camera, Microphone, and Calendar icons.
- **Status Metric**: Restored the "Last Sync" label in the session status widget.
- **Empty State Messages**: Restored the original generic "Clinical definitions will appear here" placeholder text.

> [!NOTE]
> The Cancer/Non-Cancer classification display and verdict badge introduced in earlier sessions were **retained** during this restoration — only the structural/icon changes were rolled back.

---

# Walkthrough - UI Simplification for Clinicians

The interface was refined to be more accessible to clinicians by replacing technical jargon with plain-language labels and adding key usability features.

## Changes Made in `app/templates/index.html`

| Area | Before | After |
|---|---|---|
| Metric labels | Technical model terminology | Plain English descriptions |
| Verdict display | Raw prediction string | Prominent **Cancer / Non-Cancer** badge |
| Confidence score | Raw percentage only | Percentage + band label (`High` / `Moderate` / `Low`) |
| Bottom toolbar buttons | Generic (Folder, Camera, Mic, Calendar) | Contextual actions (Copy Results, Image Info, etc.) |

## Verified UI State

![Simplified UI — idle state ready for upload](/home/moeenuddin/.gemini/antigravity/brain/19fa7ca7-b7b1-4f37-9886-b3f0c6680584/ui_simplified.png)

The dashboard now shows:
- **"Upload a lesion image to begin"** as the hero prompt
- **"AI Confidence"** with the sub-label *"How sure the AI is in its prediction"*
- **"What This Means"** panel with a plain-language explanation placeholder
- **"Analysis Status"** and **"Detailed Results"** widgets replacing raw clinical codes
/;
________()