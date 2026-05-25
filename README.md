ai_activity7/
├── training/
│   ├── prepare_dataset.py   
│   └── train_models.py      
├── streamlit_app/
│   └── app.py               
├── outputs/                 
│   ├── results.json
│   ├── MobileNetV2_model.keras
│   ├── ResNet50_model.keras
│   ├── EfficientNetB0_model.keras
│   ├── training_curves.png
│   ├── comparison_bar.png
│   └── confusion_matrices.png
└── requirements.txt
```

---

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
cd training
python prepare_dataset.py
```
This downloads the Cats vs Dogs dataset (~2,400 images) via TensorFlow Datasets
and organizes it into `dataset/train/` and `dataset/test/`.

**Alternative:** Use your own dataset:
```
dataset/
  train/
    cats/  (≥500 images)
    dogs/  (≥500 images)
  test/
    cats/  (≥100 images)
    dogs/  (≥100 images)
```

### 3. Train Models
```bash
python train_models.py
```
- Trains MobileNetV2, ResNet50, and EfficientNetB0
- Saves each model to `outputs/`
- Generates performance charts
- Prints a summary comparison table

### 4. Run the Streamlit App
```bash
cd ../streamlit_app
# Copy or symlink outputs folder
cp -r ../training/outputs ./outputs
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 🏗️ Model Architecture Summary

| Model         | ImageNet Weights | Trainable Params | Key Feature |
|---------------|:----------------:|:----------------:|-------------|
| MobileNetV2   | ✅               | 3.4M             | Depthwise separable conv, mobile-optimized |
| ResNet50      | ✅               | 25.6M            | Residual skip connections, 50 layers deep  |
| EfficientNetB0| ✅               | 5.3M             | Compound scaling (depth+width+resolution)  |

All models:
- Use ImageNet pre-trained weights (frozen base)
- Custom head: GlobalAveragePooling → BatchNorm → Dense → Dropout → Sigmoid
- Trained with Adam optimizer, binary cross-entropy loss
- Early stopping (patience=4) + ReduceLROnPlateau (patience=2)

---

## 📊 Expected Results (Approximate)

| Model          | Train Acc | Val Acc | Test Acc | Time   |
|----------------|:---------:|:-------:|:--------:|:------:|
| MobileNetV2    | ~91%      | ~88%    | ~88%     | ~5 min |
| ResNet50       | ~93%      | ~90%    | ~89%     | ~8 min |
| EfficientNetB0 | ~95%      | ~93%    | ~92%     | ~6 min |

*Results vary based on hardware and dataset size.*

---

## 🌐 Streamlit App Features

- **Model selector** — switch between all 3 models in the sidebar
- **Image upload** — drag & drop JPG/PNG/WebP
- **Real-time prediction** — shows class label, confidence %, probability bars
- **Comparison tab** — live performance table from `results.json`, training plots, analysis

---

## 📝 Analysis

**EfficientNetB0** typically achieves the best accuracy because it applies compound 
scaling — balancing network depth, width, and resolution simultaneously — achieving 
more accuracy per parameter than ResNet50 or MobileNetV2.

**ResNet50** performs well due to residual connections preventing vanishing gradients, 
but its larger parameter count (25.6M) risks overfitting on small datasets.

**MobileNetV2** is the fastest to train and most suitable for mobile/edge deployment, 
with only a slight accuracy tradeoff.
