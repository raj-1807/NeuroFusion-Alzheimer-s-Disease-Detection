# NeuroFusion AD: Multi-Modal Alzheimer's Detection

NeuroFusion AD is a Flask-based research application for Alzheimer's disease detection using multiple medical data modalities. The system combines MRI image classification, FDG PET and Tau PET image analysis, clinical feature prediction, and an attention-based fusion model to produce a final Alzheimer's risk prediction.

The project includes a lightweight HTML frontend for uploading scans and clinical data, plus a REST API backend that loads saved machine learning and deep learning models from the `models/` directory.

## Features

- MRI-based Alzheimer's probability prediction using a TensorFlow/Keras CNN model.
- Clinical-data prediction using an XGBoost pipeline with saved preprocessing artifacts.
- FDG PET and Tau PET classification using PyTorch EfficientNet-B0 models.
- Attention fusion model that combines available modality probabilities.
- Graceful partial-input support when only some modalities are provided.
- Browser-based frontend through `frontend_simple.html`.
- Health endpoint for checking model loading status.

## Project Structure

```text
Final_Project/
├── app.py
├── frontend_simple.html
├── requirements.txt
├── README.md
├── models/
│   ├── cnn_binary_model.h5
│   ├── fdg_model.pth
│   ├── tau_model.pth
│   ├── attention_fusion_best.pth
│   ├── attention_fusion_model.pth
│   ├── attention_fusion_config.json
│   ├── xgb_model.pkl
│   ├── xgb_scaler.pkl
│   ├── xgb_selector.pkl
│   ├── xgb_label_encoder.pkl
│   ├── xgb_selected_features.pkl
│   └── xgb_train_columns.pkl
├── mri/
├── fdg/
└── tau/
```

## Tech Stack

- Python
- Flask
- TensorFlow/Keras
- PyTorch
- TorchVision
- XGBoost
- scikit-learn
- NumPy
- Pillow
- Flask-CORS
- HTML, CSS, and JavaScript

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Make sure the trained model files are available inside the `models/` directory.

4. Start the Flask server.

```bash
python app.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

## Usage

1. Start the Flask backend with `python app.py`.
2. Open `frontend_simple.html` in a browser.
3. Upload one or more supported scan images:
   - MRI image
   - FDG PET image
   - Tau PET image
4. Enter or load clinical feature values.
5. Run the analysis to receive modality-level probabilities and the final fused prediction.

## API Reference

### GET `/`

Returns a simple API status message.

### GET `/health`

Returns the loading status of the saved models.

Example response:

```json
{
  "cnn_model": true,
  "xgb_pipeline": true,
  "fdg_model": true,
  "tau_model": true,
  "fusion_model": true
}
```

### POST `/predict`

Runs Alzheimer's prediction from uploaded images and/or clinical data.

Form data fields:

| Field | Type | Description |
| --- | --- | --- |
| `mri_image` | File | MRI scan image in JPG or PNG format |
| `fdg_image` | File | FDG PET scan image in JPG or PNG format |
| `tau_image` | File | Tau PET scan image in JPG or PNG format |
| `clinical_data` | String | JSON string containing clinical feature values |

Example response:

```json
{
  "cnn_prob": 0.82,
  "xgb_prob": 0.74,
  "pet_prob": 0.68,
  "fdg_probs": [0.10, 0.22, 0.68],
  "tau_probs": [0.08, 0.24, 0.68],
  "fusion_prob": 0.79,
  "fusion_type": "attention_fusion",
  "alphas": [0.67, 0.0, 0.33],
  "final_class": 1,
  "final_label": "Alzheimer's Detected",
  "threshold": 0.5
}
```

## Model Components

| Component | File | Framework | Purpose |
| --- | --- | --- | --- |
| MRI CNN | `cnn_binary_model.h5` | TensorFlow/Keras | Predicts Alzheimer's probability from MRI images |
| Clinical XGBoost | `xgb_model.pkl` and preprocessing files | XGBoost/scikit-learn | Predicts Alzheimer's probability from clinical features |
| FDG PET Model | `fdg_model.pth` | PyTorch | Classifies FDG PET scans |
| Tau PET Model | `tau_model.pth` | PyTorch | Classifies Tau PET scans |
| Attention Fusion | `attention_fusion_best.pth` or `attention_fusion_model.pth` | PyTorch | Combines modality probabilities into a final prediction |

## Notes

- The backend loads models on startup when `app.py` is run directly.
- If a fusion checkpoint cannot be loaded, the app falls back to configured mean attention weights.
- The application is intended for academic and research use. It is not a substitute for professional medical diagnosis or clinical decision-making.

## License

This project is currently provided for educational and research purposes. Add a formal license before public distribution.
