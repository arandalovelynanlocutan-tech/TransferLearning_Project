"""
Transfer Learning Image Classifier — Streamlit App (PyTorch)
Run with:  streamlit run app.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

MODELS_DIR = Path("saved_models")
IMG_SIZE   = (224, 224)

st.set_page_config(
    page_title="Transfer Learning Classifier",
    page_icon="🤖",
    layout="wide",
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_path: str):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def load_results():
    path = MODELS_DIR / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(img.convert("RGB")).unsqueeze(0)

def predict(model, img_tensor, class_names):
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    label_idx = int(prob >= 0.5)
    confidence = prob if label_idx == 1 else 1 - prob
    return class_names[label_idx], confidence, prob

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.title("🔬 Model Info")
data = load_results()

if data:
    best_name   = data["best_model"]
    class_names = data["class_names"]
    st.sidebar.success(f"**Best model:** {best_name}")
    st.sidebar.write("**Classes:**", class_names)

    import pandas as pd
    st.sidebar.subheader("📊 Performance Comparison")
    df = pd.DataFrame(data["results"])[["model", "training_acc", "val_acc", "test_acc", "training_time_s"]]
    df.columns = ["Model", "Train Acc", "Val Acc", "Test Acc", "Time (s)"]
    for col in ["Train Acc", "Val Acc", "Test Acc"]:
        df[col] = df[col].apply(lambda x: f"{x:.2%}")
    st.sidebar.dataframe(df, use_container_width=True, hide_index=True)
else:
    best_name   = None
    class_names = ["cats", "dogs"]
    st.sidebar.warning("⚠️ Run `train_models.py` first.")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
st.title("🖼️ Transfer Learning Image Classifier")
st.markdown("Upload an image and the best transfer learning model will predict its class.")

tab_predict, tab_compare, tab_about = st.tabs(["🔍 Predict", "📈 Model Comparison", "ℹ️ About"])

# ── TAB 1: Predict ──────────────────────────
with tab_predict:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        if uploaded and best_name:
            model_path = str(MODELS_DIR / f"{best_name}.pt")
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                with st.spinner("Running inference…"):
                    model      = load_model(model_path)
                    img_tensor = preprocess(img)
                    t0 = time.time()
                    label, confidence, prob = predict(model, img_tensor, class_names)
                    latency = time.time() - t0

                st.subheader("Prediction Result")
                st.markdown(f"### 🏷️ {label.upper()}")
                st.progress(confidence)
                st.markdown(f"**Confidence:** {confidence:.2%}")
                st.caption(f"Inference time: {latency*1000:.1f} ms  |  Model: {best_name}")

                st.subheader("Class Probabilities")
                probs = [1 - prob, prob]
                for cls, p in zip(class_names, probs):
                    st.markdown(f"**{cls}**")
                    st.progress(float(p))
                    st.caption(f"{p:.2%}")

        elif uploaded and not best_name:
            st.info("Please run `train_models.py` first, then restart the app.")

# ── TAB 2: Comparison ───────────────────────
with tab_compare:
    if data:
        import pandas as pd
        st.subheader("Performance Comparison Table")
        df_full = pd.DataFrame(data["results"])[["model", "training_acc", "val_acc", "test_acc", "training_time_s"]]
        df_full.columns = ["Model", "Train Acc", "Val Acc", "Test Acc", "Time (s)"]

        def highlight_best(row):
            return ["background-color: #d4edda"] * len(row) if row["Model"] == best_name else [""] * len(row)

        styled = (
            df_full.style.apply(highlight_best, axis=1)
            .format({"Train Acc": "{:.2%}", "Val Acc": "{:.2%}", "Test Acc": "{:.2%}", "Time (s)": "{:.1f}s"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        curve_path = MODELS_DIR / "training_curves.png"
        if curve_path.exists():
            st.subheader("Training Curves")
            st.image(str(curve_path), use_container_width=True)

        st.subheader("📝 Analysis")
        best_row = next(r for r in data["results"] if r["model"] == best_name)
        st.markdown(f"""
**Best model: {best_name}** achieved:
- **Validation accuracy:** {best_row['val_acc']:.2%}
- **Test accuracy:** {best_row['test_acc']:.2%}
- **Training time:** {best_row['training_time_s']:.1f} seconds

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| MobileNetV2 | Lightweight, fast | Lower capacity |
| ResNet50 | Deep residual connections | Slower, more memory |
| EfficientNetB0 | Compound scaling, best accuracy | Slightly slower than MobileNetV2 |
        """)
    else:
        st.info("Run `python train_models.py` first to generate comparison data.")

# ── TAB 3: About ────────────────────────────
with tab_about:
    st.markdown("""
## About This App

**AI 7.0 Capstone Activity** — Transfer Learning Performance Comparison

### Models Used
| Model | Parameters | ImageNet Top-1 Acc |
|-------|-----------|---------------------|
| MobileNetV2 | ~3.4 M | 71.8% |
| ResNet50 | ~25.6 M | 76.0% |
| EfficientNetB0 | ~5.3 M | 77.1% |

### Tech Stack
- **PyTorch / TorchVision** — model training & inference
- **Streamlit** — web application
- **Python 3.14**
    """)