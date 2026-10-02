# 🧠 NeuroFusion AD — Multi-Modal Alzheimer's Disease Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A research-grade multi-modal Alzheimer's disease detection system that fuses MRI, FDG PET, Tau PET, and clinical data through an attention-based deep learning pipeline.**

[Features](#-features) · [Architecture](#-architecture) · [Setup](#-setup) · [Usage](#-usage) · [API Reference](#-api-reference) · [Model Details](#-model-components)

</div>

---

## ✨ Features

- **MRI Analysis** — CNN-based Alzheimer's probability from brain MRI scans (TensorFlow/Keras)
- **FDG PET Classification** — EfficientNet-B0 fine-tuned for FDG PET scan analysis (PyTorch)
- **Tau PET Classification** — EfficientNet-B0 fine-tuned for Tau protein PET scan analysis (PyTorch)
- **Clinical Data Prediction** — XGBoost pipeline with feature selection and preprocessing (scikit-learn)
- **Attention-Based Fusion** — Soft-attention meta-learner that dynamically weighs modality probabilities
- **Graceful Partial Input** — Works even when only some modalities are provided
- **REST API** — Flask backend with health monitoring and CORS support
- **Browser Frontend** — Lightweight single-file HTML/CSS/JS interface for scan upload and analysis

## 🏗 Architecture

```
                    ┌─────────────┐
                    │  MRI Image  │
                    └──────┬──────┘
                           │ CNN (Keras)
                           ▼
                    ┌──────────────┐
                    │  P(AD|MRI)   │──────┐
                    └──────────────┘      │
                                          │
                    ┌─────────────┐       │
                    │  FDG PET    │       │
                    └──────┬──────┘       │    ┌───────────────────┐
                           │ EfficientNet │    │   Attention-Based │
                           ▼              ├───▶│   Fusion Model    │──▶ Final P(AD)
                    ┌──────────────┐      │    │   (Soft Weights)  │
                    │  P(AD|FDG)   │──────┤    └───────────────────┘
                    └──────────────┘      │
                                          │
                    ┌─────────────┐       │
                    │  Tau PET    │       │
                    └──────┬──────┘       │
                           │ EfficientNet │
                           ▼              │
                    ┌──────────────┐      │
                    │  P(AD|Tau)   │──────┤
                    └──────────────┘      │
                                          │
                    ┌─────────────┐       │
                    │Clinical Data│       │
                    └──────┬──────┘       │
                           │ XGBoost      │
                           ▼              │
                    ┌──────────────┐      │
                    │ P(AD|Clin.) │──────┘
                    └──────────────┘
```

The **Attention Fusion Model** learns dynamic weights (α₁, α₂, α₃) for each modality through a soft-attention mechanism, rather than using fixed averaging. This allows the system to emphasize the most informative modality for each individual patient.

## 📁 Project Structure

```text
NeuroFusion-Alzheimer-s-Disease-Detection/
├── app.py                      # Flask backend with all model inference logic
├── frontend_simple.html        # Browser-based frontend UI
├── requirements.txt            # Python dependencies
├── README.md
├── .gitattributes              # GitHub language detection overrides
├── .gitignore
├── models/
│   ├── cnn_binary_model.h5     # Keras CNN for MRI (tracked via Git LFS)
│   ├── fdg_model.pth           # EfficientNet-B0 for FDG PET
│   ├── tau_model.pth           # EfficientNet-B0 for Tau PET
│   ├── attention_fusion_best.pth
│   ├── attention_fusion_model.pth
│   ├── attention_fusion_config.json
│   ├── xgb_model.pkl           # XGBoost classifier
│   ├── xgb_scaler.pkl          # Feature scaler
│   ├── xgb_selector.pkl        # Feature selector
│   ├── xgb_label_encoder.pkl
│   ├── xgb_selected_features.pkl
│   └── xgb_train_columns.pkl
├── mri/                        # Sample MRI scan images (OASIS dataset)
├── fdg/                        # Sample FDG PET scan images
└── tau/                        # Sample Tau PET scan images
```

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| **Backend** | Python, Flask, Flask-CORS |
| **Deep Learning** | TensorFlow/Keras, PyTorch, TorchVision |
| **Machine Learning** | XGBoost, scikit-learn |
| **Data Processing** | NumPy, Pandas, Pillow |
| **Frontend** | HTML5, CSS3, JavaScript (vanilla) |
| **Model Architecture** | EfficientNet-B0, Custom CNN, Soft-Attention Fusion |

## 🚀 Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git LFS (for the large CNN model file)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/raj-1807/NeuroFusion-Alzheimer-s-Disease-Detection.git
   cd NeuroFusion-Alzheimer-s-Disease-Detection
   ```

2. **Install Git LFS and pull large files**

   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create and activate a virtual environment**

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

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Verify model files are present** in the `models/` directory.

6. **Start the Flask server**

   ```bash
   python app.py
   ```

   The backend runs at `http://127.0.0.1:5000`

## 📖 Usage

1. Start the Flask backend with `python app.py`.
2. Open `frontend_simple.html` in a browser.
3. Upload one or more supported scan images:
   - **MRI image** (JPG/PNG brain scan)
   - **FDG PET image** (PNG PET scan)
   - **Tau PET image** (PNG PET scan)
4. Enter or load clinical feature values.
5. Run the analysis to receive modality-level probabilities and the final fused prediction.

## 📡 API Reference

### `GET /`

Returns a simple API status message.

### `GET /health`

Returns the loading status of all models.

```json
{
  "cnn_model": true,
  "xgb_pipeline": true,
  "fdg_model": true,
  "tau_model": true,
  "fusion_model": true
}
```

### `POST /predict`

Runs Alzheimer's prediction from uploaded images and/or clinical data.

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| `mri_image` | File | MRI scan image (JPG/PNG) |
| `fdg_image` | File | FDG PET scan image (JPG/PNG) |
| `tau_image` | File | Tau PET scan image (JPG/PNG) |
| `clinical_data` | String | JSON string with clinical feature values |

**Response Example:**

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

## 🧩 Model Components

| Component | File | Framework | Purpose |
|-----------|------|-----------|---------|
| **MRI CNN** | `cnn_binary_model.h5` | TensorFlow/Keras | Binary classification from MRI images |
| **Clinical XGBoost** | `xgb_model.pkl` + preprocessing | XGBoost/scikit-learn | Prediction from clinical tabular features |
| **FDG PET Model** | `fdg_model.pth` | PyTorch (EfficientNet-B0) | 3-class FDG PET scan classification |
| **Tau PET Model** | `tau_model.pth` | PyTorch (EfficientNet-B0) | 3-class Tau PET scan classification |
| **Attention Fusion** | `attention_fusion_best.pth` | PyTorch | Soft-attention weighted fusion of modalities |

## ⚠️ Important Notes

- **Academic Use Only** — This system is intended for research and educational purposes. It is **not** a substitute for professional medical diagnosis or clinical decision-making.
- The backend loads all models on startup when `app.py` runs.
- If a fusion checkpoint cannot be loaded, the app falls back to configured mean attention weights.
- The CNN model file (`cnn_binary_model.h5`, ~112 MB) is tracked with **Git LFS**. Run `git lfs pull` after cloning.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ for Alzheimer's research</sub>
</div>
