# ✏️ SmartBoard AI

> **Handwritten Math Learning Assistant** — draw digits and equations, get instant recognition, step-by-step solutions, and handwriting quality feedback.

![SmartBoard AI Banner](docs/banner-placeholder.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SmartBoard AI                         │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐  ┌─────────────┐  │
│  │   Streamlit  │   │   Drawing    │  │   Upload    │  │
│  │   Frontend   │◄──│   Canvas     │  │   Image     │  │
│  └──────┬───────┘   └──────────────┘  └─────────────┘  │
│         │                                               │
│  ┌──────▼───────────────────────────────────────────┐   │
│  │              preprocess.py (OpenCV + Pillow)      │   │
│  │  • Adaptive thresholding                          │   │
│  │  • Contour detection & bounding boxes             │   │
│  │  • Box merging (handles multi-stroke chars)       │   │
│  │  • 28×28 MNIST-style patch extraction             │   │
│  └──────┬───────────────────────┬───────────────────┘   │
│         │                       │                       │
│  ┌──────▼───────┐       ┌───────▼──────┐               │
│  │  model.py    │       │  quality.py  │               │
│  │  CNN (Keras) │       │  CV heuristics│              │
│  │  MNIST ~99%  │       │  0-100 score  │              │
│  └──────┬───────┘       └───────┬──────┘               │
│         │                       │                       │
│  ┌──────▼───────────────────────▼──────────────────┐   │
│  │              solver.py (SymPy)                   │   │
│  │  • Tokenize label sequence → expression          │   │
│  │  • Parse with implicit multiplication            │   │
│  │  • Arithmetic evaluation + symbolic simplify     │   │
│  │  • Linear / quadratic equation solving           │   │
│  │  • Step-by-step LaTeX output                     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
smartboard_ai/
├── app.py              # Main Streamlit UI (tabs: Draw, Upload, Examples, History)
├── model.py            # CNN architecture, training loop, inference utilities
├── preprocess.py       # OpenCV pipeline — binarise, segment, extract patches
├── solver.py           # SymPy expression parser + step-by-step solver
├── quality.py          # Handwriting quality scorer (CV heuristics, 5 metrics)
├── requirements.txt    # Pinned Python dependencies
├── README.md           # This file
└── smartboard_model.h5 # Auto-generated after first run
```

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/smartboard-ai.git
cd smartboard-ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

On first launch the app trains the CNN on MNIST (~2 min on CPU). The model is saved to `smartboard_model.h5` and loaded instantly on subsequent runs.

### 3. Use

| Tab | What you do |
|-----|-------------|
| **Draw** | Draw a digit or expression on the canvas → click **Analyse** |
| **Upload** | Upload a photo or scan of handwritten math |
| **Examples** | Browse and solve pre-loaded expressions interactively |
| **History** | Review all solved expressions + quality trend chart |

---

## 🧠 ML Details

### CNN Architecture

| Layer | Details |
|-------|---------|
| Input | 28 × 28 × 1 (greyscale) |
| Conv Block 1 | 2 × Conv2D(32, 3×3, ReLU) + BN + MaxPool + Dropout(0.25) |
| Conv Block 2 | 2 × Conv2D(64, 3×3, ReLU) + BN + MaxPool + Dropout(0.25) |
| Conv Block 3 | Conv2D(128, 3×3, ReLU) + BN + Dropout(0.25) |
| Head | GlobalAvgPool → Dense(256, ReLU) + BN + Dropout(0.5) → Softmax(10) |

- **Dataset**: MNIST (60 000 train / 10 000 test)
- **Optimiser**: Adam (lr=1e-3, ReduceLROnPlateau)
- **Achieved accuracy**: ~99.3 % on test set

### Handwriting Quality Metrics

| Metric | Method | Weight |
|--------|--------|--------|
| Stroke consistency | Distance-transform CV | 30 % |
| Ink coverage | Ink-to-canvas density | 20 % |
| Alignment | Hough line count | 15 % |
| Cleanliness | Small-component count | 25 % |
| Proportions | Bounding box aspect ratio | 10 % |

### Segmentation Pipeline

```
Canvas PNG
    │
    ▼
Greyscale → Adaptive Threshold → Morphological Close
    │
    ▼
FindContours (RETR_EXTERNAL)
    │
    ▼
Filter noise (area < 100 px²)
    │
    ▼
Iterative box merge (gap ≤ 15 px) → handles multi-stroke digits
    │
    ▼
Sort left-to-right → extract 28×28 MNIST-normalised patches
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push repo to GitHub (include `requirements.txt` and all `.py` files)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, main file `app.py`
4. Under **Advanced settings** → add secret if needed (none required for base version)
5. Click **Deploy** — Streamlit Cloud handles the pip install and model training automatically

> **Tip:** To avoid cold-start re-training, commit a pre-trained `smartboard_model.h5` to the repo (file size ~2 MB for the default architecture).

---

## 🖼️ Screenshots

| Draw Tab | Solution Panel |
|----------|----------------|
| ![Draw](docs/draw-placeholder.png) | ![Solution](docs/solution-placeholder.png) |

---

## 🌟 ML Portfolio Highlights

This project demonstrates:

- **End-to-end ML pipeline** — data loading, preprocessing, training, evaluation, and inference in a single codebase
- **Production patterns** — model caching with `@st.cache_resource`, graceful fallbacks for low confidence, error handling throughout
- **Computer vision** — adaptive thresholding, contour detection, morphological operations, and distance transforms
- **Symbolic AI** — SymPy for exact algebraic solving with LaTeX output (not just numeric approximations)
- **Interpretable ML** — handwriting quality decomposed into 5 explainable CV metrics with actionable user tips
- **Full-stack deployment** — Streamlit + Streamlit Cloud with zero additional infrastructure

---

## 🔭 3 Extensions to Make It Stand Out

### 1. 🎨 Data Augmentation + Custom Symbol Training
Extend the dataset with synthetic operator images ('+', '-', '×', '÷', '=', parentheses) generated via `albumentations`:
```python
# augment.py
from albumentations import Compose, Rotate, ElasticTransform, GaussNoise
pipeline = Compose([Rotate(limit=15), ElasticTransform(alpha=1, sigma=50), GaussNoise()])
```
Train a 17-class model (10 digits + 7 symbols) to recognise full equations end-to-end without the manual symbol mapping.

### 2. 📷 Real-time Webcam Mode (OpenCV + Streamlit)
Add a webcam tab using `streamlit-webrtc` to recognise digits drawn on paper in real time:
- Frame capture → crop centre ROI → greyscale → thresholding → CNN inference
- Running average of last 5 frames to smooth predictions
- This turns the project into an AR-style live demo — very impressive in portfolio videos

### 3. 🧮 Calculus & Trigonometry Solver
Extend `solver.py` with SymPy's `diff`, `integrate`, and `trigsimp`:
```python
import sympy as sp
x = sp.Symbol('x')
# Differentiation
sp.diff(sp.sin(x)**2 + x**3, x)   # → 2*sin(x)*cos(x) + 3*x**2
# Integration
sp.integrate(x**2, x)              # → x**3/3
```
Pair with `matplotlib` to render function plots inline — turns SmartBoard AI into a full symbolic calculator with educational graph output.

---

## 📄 License

MIT — free to use, modify, and deploy.
