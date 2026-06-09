"""
Alzheimer's Multi-Modal Detection Backend
Flask API serving: CNN-MRI + XGBoost-Clinical + EfficientNet-PET + Attention Fusion
"""

import os
import json
import pickle
import logging
import warnings
import traceback
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CNN_IMG_SIZE   = (224, 224)
PET_IMG_SIZE   = (224, 224)

# Loaded model handles (populated in load_all_models)
_cnn_model    = None   # Keras
_xgb_pipeline = None   # dict of sklearn/xgb objects
_fdg_model    = None   # PyTorch
_tau_model    = None   # PyTorch
_fusion_model = None   # PyTorch  OR  dict (fallback with mean_alpha)
_fusion_cfg   = {}

# ─────────────────────────────────────────────
# MODEL DEFINITIONS  (PyTorch)
# ─────────────────────────────────────────────
def _build_pet_model(num_classes=3):
    """EfficientNet-B0 fine-tuned for 3-class PET classification."""
    import torch.nn as nn
    import torchvision.models as tv

    model = tv.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features          # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class AttentionFusionModel:
    """
    Soft-attention meta-learner over 3 modality scalar probabilities.

    Matches the saved checkpoint architecture:
        attention_net : Sequential(
            Linear(3 → hidden_dim), ReLU, Dropout,
            Linear(hidden_dim → hidden_dim), ReLU,
            Linear(hidden_dim → 3), Softmax            → alpha weights
        )
        output_layer : Sequential(
            Linear(1 → 8), ReLU,
            Linear(8 → 1)                              → sigmoid → P(AD)
        )
    """
    def __init__(self, n_modalities=3, hidden_dim=32):
        import torch.nn as nn

        self.n_mod  = n_modalities
        self.hidden = hidden_dim

        # ── attention_net  (keys: attention_net.0/3/5) ──
        self.attention_net = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),   # .0
            nn.ReLU(),                              # .1
            nn.Dropout(p=0.3),                      # .2
            nn.Linear(hidden_dim, hidden_dim),      # .3
            nn.ReLU(),                              # .4
            nn.Linear(hidden_dim, n_modalities),    # .5
            nn.Softmax(dim=-1),                     # .6
        )

        # ── output_layer  (keys: output_layer.0/2) ──
        self.output_layer = nn.Sequential(
            nn.Linear(1, 8),                        # .0
            nn.ReLU(),                              # .1
            nn.Linear(8, 1),                        # .2
        )

    def parameters(self):
        import itertools
        return itertools.chain(
            self.attention_net.parameters(),
            self.output_layer.parameters(),
        )

    def to(self, device):
        self.attention_net = self.attention_net.to(device)
        self.output_layer  = self.output_layer.to(device)
        return self

    def eval(self):
        self.attention_net.eval()
        self.output_layer.eval()
        return self

    def __call__(self, x):
        """
        x : (B, 3) – three modality probabilities.
        Returns (prob, alpha)  where prob is (B, 1) sigmoid output.
        """
        import torch
        alpha  = self.attention_net(x)                # (B, 3)
        fused  = (alpha * x).sum(dim=-1, keepdim=True)  # weighted sum → (B, 1)
        logit  = self.output_layer(fused)             # (B, 1)
        prob   = torch.sigmoid(logit)                 # (B, 1)
        return prob, alpha

    def load_state_dict(self, sd, strict=True):
        """Load state-dict with matching key names."""
        import torch.nn as nn

        # Build a temporary nn.Module so we can use PyTorch's native loader
        class _Wrapper(nn.Module):
            def __init__(self, attn, out):
                super().__init__()
                self.attention_net = attn
                self.output_layer = out

        wrapper = _Wrapper(self.attention_net, self.output_layer)
        wrapper.load_state_dict(sd, strict=strict)
        self.attention_net = wrapper.attention_net
        self.output_layer  = wrapper.output_layer
        log.info("AttentionFusionModel loaded %d tensor(s).", len(sd))


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
def _path(filename):
    return os.path.join(MODELS_DIR, filename)


def _load_cnn():
    global _cnn_model
    try:
        import tensorflow as tf
        _cnn_model = tf.keras.models.load_model(_path("cnn_binary_model.h5"), compile=False)
        log.info("✅ CNN-MRI model loaded (Keras).")
    except Exception as e:
        log.warning("⚠️  CNN-MRI model failed to load: %s", e)


def _load_xgb_pipeline():
    global _xgb_pipeline
    try:
        import joblib

        train_cols   = joblib.load(_path("xgb_train_columns.pkl"))
        sel_features = joblib.load(_path("xgb_selected_features.pkl"))

        # Scaler / selector may have version issues – try loading with fallback
        try:
            scaler = joblib.load(_path("xgb_scaler.pkl"))
        except Exception as e:
            log.warning("xgb_scaler load warning: %s", e)
            scaler = None

        try:
            selector = joblib.load(_path("xgb_selector.pkl"))
        except Exception as e:
            log.warning("xgb_selector load warning: %s", e)
            selector = None

        try:
            le = joblib.load(_path("xgb_label_encoder.pkl"))
        except Exception as e:
            log.warning("xgb_label_encoder load warning: %s", e)
            le = None

        xgb_model = joblib.load(_path("xgb_model.pkl"))

        _xgb_pipeline = {
            "model":         xgb_model,
            "train_cols":    train_cols,
            "sel_features":  sel_features,
            "scaler":        scaler,
            "selector":      selector,
            "label_encoder": le,
        }
        log.info("✅ XGBoost pipeline loaded.")
    except Exception as e:
        log.warning("⚠️  XGBoost pipeline failed to load: %s", e)
        traceback.print_exc()


def _load_pet_models():
    global _fdg_model, _tau_model
    import torch

    device = torch.device("cpu")

    for attr, fname in [("_fdg_model", "fdg_model.pth"), ("_tau_model", "tau_model.pth")]:
        path = _path(fname)
        try:
            # Try full-model load first (most likely how they were saved)
            obj = torch.load(path, map_location=device, weights_only=False)
            if hasattr(obj, "eval"):
                obj.eval()
                globals()[attr] = obj
                log.info("✅ %s loaded as full model.", fname)
            elif isinstance(obj, dict):
                # It's a state dict; build architecture and load
                m = _build_pet_model(num_classes=3)
                m.load_state_dict(obj, strict=True)
                m.eval()
                globals()[attr] = m
                log.info("✅ %s loaded from state-dict.", fname)
            else:
                log.warning("⚠️  %s: unknown object type %s", fname, type(obj))
        except Exception as e:
            log.warning("⚠️  %s failed to load: %s", fname, e)
            traceback.print_exc()


def _load_fusion_model():
    global _fusion_model, _fusion_cfg
    import torch

    # Load config
    try:
        with open(_path("attention_fusion_config.json")) as f:
            _fusion_cfg = json.load(f)
        log.info("Fusion config: %s", _fusion_cfg)
    except Exception as e:
        log.warning("Could not load fusion config: %s", e)
        _fusion_cfg = {"n_modalities": 3, "hidden_dim": 32, "mean_alpha": [0.67, 0.0001, 0.33]}

    n_mod   = _fusion_cfg.get("n_modalities", 3)
    h_dim   = _fusion_cfg.get("hidden_dim", 32)

    for fname in ["attention_fusion_best.pth", "attention_fusion_model.pth"]:
        path = _path(fname)
        if not os.path.exists(path):
            continue
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)

            if hasattr(obj, "eval"):
                # Saved as full model
                obj.eval()
                _fusion_model = obj
                log.info("✅ Fusion model loaded as full model (%s).", fname)
                return

            if isinstance(obj, dict):
                # State dict – try our architecture
                fusion = AttentionFusionModel(n_modalities=n_mod, hidden_dim=h_dim)
                try:
                    fusion.load_state_dict(obj, strict=False)
                    fusion.eval()
                    _fusion_model = fusion
                    log.info("✅ Fusion model loaded from state-dict (%s).", fname)
                    return
                except Exception as inner:
                    log.warning("State-dict load attempt failed: %s – will try next file.", inner)

        except Exception as e:
            log.warning("⚠️  %s fusion load error: %s", fname, e)

    # Fallback: use mean_alpha weighted average (no neural net needed)
    log.warning("⚠️  Using mean_alpha fallback fusion (no neural-net loaded).")
    _fusion_model = {"fallback": True, "mean_alpha": _fusion_cfg.get("mean_alpha", [1/3, 1/3, 1/3])}


def load_all_models():
    log.info("Loading all models from %s …", MODELS_DIR)
    _load_cnn()
    _load_xgb_pipeline()
    _load_pet_models()
    _load_fusion_model()
    log.info("Model loading complete.")


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def _preprocess_image_keras(file_obj, size=CNN_IMG_SIZE):
    """Load image → (1, H, W, 3) float32 normalised to [0,1]."""
    img = Image.open(file_obj).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]          # (1, H, W, 3)


def _preprocess_image_torch(file_obj, size=PET_IMG_SIZE):
    """Load image → (1, 3, H, W) torch.Tensor (ImageNet normalised)."""
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img    = Image.open(file_obj).convert("RGB")
    tensor = tfm(img).unsqueeze(0)       # (1, 3, H, W)
    return tensor


def run_cnn_mri(file_storage):
    """Returns scalar P(Alzheimer) from CNN-MRI model."""
    if _cnn_model is None:
        log.warning("CNN model is None – skipping MRI inference.")
        return None
    try:
        arr    = _preprocess_image_keras(BytesIO(file_storage.read()))
        log.info("CNN input shape: %s, min=%.4f, max=%.4f", arr.shape, arr.min(), arr.max())
        preds  = _cnn_model.predict(arr, verbose=0)   # (1, 1) or (1, 2)
        log.info("CNN raw output: %s (type=%s)", preds, type(preds))
        preds  = np.array(preds).ravel()
        log.info("CNN flattened preds: %s  shape=%s", preds, preds.shape)
        if preds.shape[0] == 1:          # sigmoid output
            result = float(preds[0])
        else:                            # softmax output – P(class-1 = AD)
            result = float(preds[1])
        log.info("CNN final probability: %.6f", result)
        return result
    except Exception as e:
        log.error("CNN-MRI inference error: %s", e); traceback.print_exc()
        return None


def run_xgboost(clinical_dict):
    """Returns scalar P(Alzheimer) from XGBoost pipeline."""
    if _xgb_pipeline is None:
        return None
    try:
        import pandas as pd

        pipe = _xgb_pipeline

        # Build DataFrame from raw fields
        df = pd.DataFrame([clinical_dict])

        # ── Feature engineering (matches training) ──────────────────────
        df["Age_squared"]        = df["Age"] ** 2
        df["Age_group"]          = pd.cut(df["Age"], bins=[0, 60, 70, 80, 200],
                                           labels=[0, 1, 2, 3]).astype(float)
        df["MMSE_deficit"]       = (df["MMSE"] < 24).astype(int)
        df["MMSE_severe"]        = (df["MMSE"] < 10).astype(int)
        df["MMSE_mild"]          = ((df["MMSE"] >= 18) & (df["MMSE"] < 24)).astype(int)
        df["Age_MMSE_ratio"]     = df["Age"] / (df["MMSE"] + 1e-6)
        df["Age_x_MMSE_deficit"] = df["Age"] * df["MMSE_deficit"]
        df["BMI_category"]       = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 200],
                                           labels=[0, 1, 2, 3]).astype(float)
        df["Obese"]              = (df["BMI"] >= 30).astype(int)
        df["LowActivity"]        = (df["PhysicalActivity"] < 3).astype(int)
        df["Functional_x_ADL"]  = df["FunctionalAssessment"] * df["ADL"]

        # Add missing cols (PatientID dummy)
        if "PatientID" not in df.columns:
            df["PatientID"] = 0

        # Align to training column order
        train_cols   = pipe["train_cols"]
        sel_features = pipe["sel_features"]

        df = df.reindex(columns=train_cols, fill_value=0)

        # Keep only selected features
        df_sel = df[sel_features]

        # Scale
        if pipe["scaler"] is not None:
            try:
                arr = pipe["scaler"].transform(df_sel.values)
            except Exception:
                arr = df_sel.values.astype(np.float32)
        else:
            arr = df_sel.values.astype(np.float32)

        # Feature selection (if selector present)
        if pipe["selector"] is not None:
            try:
                arr = pipe["selector"].transform(arr)
            except Exception:
                pass

        # Predict
        prob = pipe["model"].predict_proba(arr)[0]
        # prob is (n_classes,) – class 1 = Alzheimer
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    except Exception as e:
        log.error("XGBoost inference error: %s", e); traceback.print_exc()
        return None


def run_pet_model(model, file_storage):
    """
    Run FDG or Tau EfficientNet model.
    Returns (prob_ad: float, probs_all: list[float]) with 3-class softmax.
    """
    if model is None:
        return None, None
    try:
        import torch
        import torch.nn.functional as F

        tensor = _preprocess_image_torch(BytesIO(file_storage.read()))
        with torch.no_grad():
            logits = model(tensor)              # (1, 3)
            probs  = F.softmax(logits, dim=1)   # (1, 3)
        probs_np  = probs.squeeze(0).numpy()    # [p_CN, p_MCI, p_AD]
        prob_ad   = float(probs_np[2])          # AD is class-2
        return prob_ad, probs_np.tolist()
    except Exception as e:
        log.error("PET inference error: %s", e); traceback.print_exc()
        return None, None


def run_attention_fusion(p_mri, p_xgb, p_pet):
    """
    Fuse available modality probabilities.
    Returns (final_prob, fusion_type, alphas).

    Neural attention weights are used when ALL 3 modalities are present
    (the attention_net was trained that way). The output_layer is deliberately
    skipped because its learned biases can distort the weighted sum and
    flip predictions inconsistently (e.g. adding clinical data to an
    otherwise-clear MRI+PET scan could wrongly trigger AD detection).

    Instead, we use the attention-weighted sum directly as the probability,
    which is consistent with the weighted-average path used for partial
    modalities.
    """
    probs_raw  = [p_mri, p_xgb, p_pet]
    available  = [p for p in probs_raw if p is not None]
    mask       = np.array([1 if p is not None else 0 for p in probs_raw], dtype=np.float64)
    n_avail    = int(mask.sum())

    if n_avail == 0:
        return 0.5, "no_data", [0.0, 0.0, 0.0]

    # If only 1 modality is available, just return its probability directly
    if n_avail == 1:
        single_prob = available[0]
        alphas = mask.tolist()            # e.g. [0, 1, 0]
        return float(single_prob), "single_modality", alphas

    # ── Neural attention weights – ONLY when ALL 3 modalities present ──
    #    We use attention_net to get dynamic alpha weights, then compute a
    #    simple weighted sum (skipping output_layer to avoid bias distortion).
    if n_avail == 3 and _fusion_model is not None and not isinstance(_fusion_model, dict):
        try:
            import torch

            x = torch.tensor([probs_raw], dtype=torch.float32)
            with torch.no_grad():
                # Get attention weights from the neural network
                if hasattr(_fusion_model, 'attention_net'):
                    alpha_t = _fusion_model.attention_net(x)   # (1, 3) softmax weights
                else:
                    # Full model – run normally but only take alphas
                    result = _fusion_model(x)
                    if isinstance(result, (tuple, list)):
                        _, alpha_t = result
                    else:
                        alpha_t = None

            if alpha_t is not None:
                alphas = alpha_t.squeeze(0).tolist()

                # Compute attention-weighted sum directly (no output_layer)
                probs_arr  = np.array(probs_raw, dtype=np.float64)
                alpha_arr  = np.array(alphas,    dtype=np.float64)
                fused_prob = float(np.dot(alpha_arr, probs_arr))

                # ── Majority-vote safety check ──────────────────────────
                #    If 2+ individual modalities agree on a class but the
                #    fused result disagrees, fall through to weighted avg.
                votes_ad   = sum(1 for p in probs_raw if p is not None and p >= 0.5)
                fused_cls  = 1 if fused_prob >= 0.5 else 0
                majority   = 1 if votes_ad >= 2 else 0

                if fused_cls != majority:
                    log.warning(
                        "Attention fusion (%.3f → class %d) disagrees with "
                        "majority vote (%d/3 predict AD). Falling back to "
                        "weighted average.",
                        fused_prob, fused_cls, votes_ad,
                    )
                    # Fall through to weighted average below
                else:
                    log.info(
                        "Attention fusion: alphas=[%.3f, %.3f, %.3f] → "
                        "weighted sum=%.4f",
                        alphas[0], alphas[1], alphas[2], fused_prob,
                    )
                    return fused_prob, "attention_fusion", alphas

        except Exception as e:
            log.warning("Neural fusion failed (%s) – falling back to weighted avg.", e)

    # ── Weighted average (partial or full) ────────────────────────────
    alpha = np.array(_fusion_cfg.get("mean_alpha", [1/3, 1/3, 1/3]), dtype=np.float64)

    # Zero-out missing modalities and re-normalise
    eff_alpha = alpha * mask
    if eff_alpha.sum() > 0:
        eff_alpha /= eff_alpha.sum()
    else:
        eff_alpha = mask / mask.sum()

    # Fill missing with 0 for dot-product (masked out by eff_alpha anyway)
    probs_safe = np.array([p if p is not None else 0.0 for p in probs_raw], dtype=np.float64)
    fused = float(np.dot(eff_alpha, probs_safe))

    # ── Final majority-vote guard (for weighted-avg path too) ─────────
    votes_ad  = sum(1 for p in probs_raw if p is not None and p >= 0.5)
    fused_cls = 1 if fused >= 0.5 else 0
    majority  = 1 if votes_ad >= 2 else 0

    if fused_cls != majority and n_avail >= 2:
        # Override: use simple equal-weight average of available modalities
        equal_avg = float(np.mean([p for p in probs_raw if p is not None]))
        log.warning(
            "Weighted avg (%.3f → class %d) disagrees with majority vote "
            "(%d/%d predict AD). Overriding with equal avg=%.3f.",
            fused, fused_cls, votes_ad, n_avail, equal_avg,
        )
        eff_alpha = mask / mask.sum()
        fused = equal_avg

    return fused, "weighted_avg", eff_alpha.tolist()


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    return send_file(os.path.join(BASE_DIR, "frontend_simple.html"))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "cnn_loaded":    _cnn_model    is not None,
        "xgb_loaded":    _xgb_pipeline is not None,
        "fdg_loaded":    _fdg_model    is not None,
        "tau_loaded":    _tau_model    is not None,
        "fusion_loaded": _fusion_model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        files   = request.files
        form    = request.form
        mri_f   = files.get("mri_image")
        fdg_f   = files.get("fdg_image")
        tau_f   = files.get("tau_image")
        clin_s  = form.get("clinical_data")

        # ── Validate at least one input ──────────────────────────────
        if not mri_f and not fdg_f and not tau_f and not clin_s:
            return jsonify({"error": "No input provided. Upload at least one image or clinical data."}), 400

        results = {}
        modalities_provided = []

        # ── 1. CNN MRI ───────────────────────────────────────────────
        cnn_prob = None
        if mri_f:
            log.info("Running CNN-MRI on %s …", mri_f.filename)
            cnn_prob = run_cnn_mri(mri_f)
            modalities_provided.append("mri")
        results["cnn_prob"] = cnn_prob           # always present (null if not provided)

        # ── 2. XGBoost Clinical ──────────────────────────────────────
        xgb_prob = None
        if clin_s:
            log.info("Running XGBoost on clinical data …")
            clinical = json.loads(clin_s)
            xgb_prob = run_xgboost(clinical)
            modalities_provided.append("clinical")
        results["xgb_prob"] = xgb_prob           # always present

        # ── 3. PET FDG + Tau ─────────────────────────────────────────
        pet_prob = None
        fdg_probs, tau_probs = None, None

        if fdg_f:
            log.info("Running FDG-PET model on %s …", fdg_f.filename)
            fdg_ad, fdg_probs = run_pet_model(_fdg_model, fdg_f)
        else:
            fdg_ad = None

        if tau_f:
            log.info("Running Tau-PET model on %s …", tau_f.filename)
            tau_ad, tau_probs = run_pet_model(_tau_model, tau_f)
        else:
            tau_ad = None

        # Combine FDG + Tau → single PET probability (average if both present)
        if fdg_ad is not None and tau_ad is not None:
            pet_prob = (fdg_ad + tau_ad) / 2.0
        elif fdg_ad is not None:
            pet_prob = fdg_ad
        elif tau_ad is not None:
            pet_prob = tau_ad

        if pet_prob is not None:
            modalities_provided.append("pet")

        results["pet_prob"]  = pet_prob            # always present
        results["fdg_probs"] = fdg_probs           # [CN, MCI, AD] or null
        results["tau_probs"] = tau_probs
        results["modalities_provided"] = modalities_provided

        # ── 4. Attention Fusion ───────────────────────────────────────
        fusion_prob, fusion_type, alphas = run_attention_fusion(cnn_prob, xgb_prob, pet_prob)
        results["fusion_prob"] = fusion_prob
        results["fusion_type"] = fusion_type
        results["alphas"]      = alphas

        # ── 5. Final Decision ─────────────────────────────────────────
        THRESHOLD = 0.5
        final_class = 1 if fusion_prob >= THRESHOLD else 0
        if final_class == 1:
            final_label = "⚠️ Alzheimer's Detected"
        else:
            final_label = "✅ No Alzheimer's Detected"

        results["final_class"] = final_class
        results["final_label"] = final_label
        results["threshold"]   = THRESHOLD

        log.info("Prediction: %s | cnn=%.3f xgb=%.3f pet=%.3f → fusion=%.3f (%s)",
                 final_label,
                 cnn_prob or -1, xgb_prob or -1, pet_prob or -1,
                 fusion_prob, fusion_type)

        return jsonify(results)

    except Exception as e:
        log.error("Prediction endpoint error: %s", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, threading
    load_all_models()
    # Auto-open browser after server starts
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
