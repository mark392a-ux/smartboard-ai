# ✏️ SmartBoard AI

> **Handwritten Math Learning Assistant** — draw digits and equations, get instant CNN recognition, step-by-step SymPy solutions, and handwriting quality feedback powered by computer vision.

<br>

<!-- ============================================================
     BANNER SCREENSHOT
     Replace the block below with your banner image:
     ![SmartBoard AI](docs/screenshots/banner.png)
     Recommended: full-width screenshot of the app (1400×700px)
     ============================================================ -->

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│          [ BANNER SCREENSHOT — add yours here ]            │
│                                                             │
│   Suggested: full-width screenshot of the Draw tab         │
│   Size: 1400 × 700 px  |  File: docs/screenshots/banner.png│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

<br>

---

## 📸 Screenshots

### Draw Tab — Live Canvas + Sidebar

<!-- Replace with: ![Draw Tab](docs/screenshots/draw-tab.png) -->
```
┌──────────────────────────────────────────────────────────────┐
│  [ SCREENSHOT: Draw tab ]                                    │
│                                                              │
│  Show: canvas with a digit/equation drawn,                   │
│        right sidebar with prediction + confidence bars       │
│  Size: 1200 × 650 px                                         │
└──────────────────────────────────────────────────────────────┘
```

<br>

### Step-by-Step Solution Panel

<!-- Replace with: ![Solution Panel](docs/screenshots/solution.png) -->
```
┌──────────────────────────────────────────────────────────────┐
│  [ SCREENSHOT: Solution panel ]                              │
│                                                              │
│  Show: equation like 2x+6=10 solved step by step            │
│        with LaTeX rendering and the Answer box              │
│  Size: 1200 × 450 px                                         │
└──────────────────────────────────────────────────────────────┘
```

<br>

### Handwriting Quality Score

<!-- Replace with: ![Quality Score](docs/screenshots/quality.png) -->
```
┌──────────────────────────────────────────────────────────────┐
│  [ SCREENSHOT: Quality sidebar card ]                        │
│                                                              │
│  Show: score ring (e.g. 84/100 Grade B), breakdown bars,    │
│        and the feedback tips below                           │
│  Size: 500 × 550 px                                          │
└──────────────────────────────────────────────────────────────┘
```

<br>

### Examples & History Tabs

<!-- Replace with: ![Examples](docs/screenshots/examples.png) -->
```
┌──────────────────────────────────────────────────────────────┐
│  [ SCREENSHOT: Examples tab ]                                │
│                                                              │
│  Show: the card grid of example expressions with            │
│        a solved result expanded below                        │
│  Size: 1200 × 500 px                                         │
└──────────────────────────────────────────────────────────────┘
```

<br>

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

---

## 📁 Project Structure

```
smartboard_ai/
├── app.py                  # Main Streamlit UI (Draw · Upload · Examples · History)
├── model.py                # CNN architecture, training loop, inference utilities
├── preprocess.py           # OpenCV pipeline — binarise, segment, extract patches
├── solver.py               # SymPy expression parser + step-by-step solver
├── quality.py              # Handwriting quality scorer (CV heuristics, 5 metrics)
├── requirements.txt        # Pinned Python dependencies
├── .python-version         # Pins Python 3.11 for Streamlit Cloud
├── README.md               # This file
└── smartboard_model.h5     # Auto-generated on first run (~2 MB)
```

---

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

On first launch the app trains the CNN on MNIST (~2 min on CPU). The model is saved to `smartboard_model.h5` and loaded instantly on all subsequent runs.

### 3. Use

| Tab | What you do |
|-----|-------------|
| **✏️ Draw** | Draw a digit or equation on the canvas → click **Analyse** |
| **📁 Upload** | Upload a photo or scan of handwritten math |
| **💡 Examples** | Browse pre-loaded expressions and click **Solve** |
| **📋 History** | Review all solved expressions + quality trend chart |

---

## 🧠 ML Details

### CNN Architecture

| Layer | Details |
|-------|---------|
| Input | 28 × 28 × 1 (greyscale) |
| Conv Block 1 | 2 × Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool + Dropout(0.25) |
| Conv Block 2 | 2 × Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool + Dropout(0.25) |
| Conv Block 3 | Conv2D(128, 3×3, ReLU) + BatchNorm + Dropout(0.25) |
| Head | GlobalAvgPool → Dense(256, ReLU) + BatchNorm + Dropout(0.5) → Softmax(10) |

- **Dataset:** MNIST — 60,000 train / 10,000 test
- **Optimiser:** Adam (lr=1e-3) with `ReduceLROnPlateau`
- **Achieved accuracy:** ~99.3% on test set

### Handwriting Quality Metrics

| Metric | Method | Weight |
|--------|--------|--------|
| Stroke consistency | Distance-transform coefficient of variation | 30% |
| Ink coverage | Ink-to-canvas pixel density | 20% |
| Alignment | Hough line confidence count | 15% |
| Cleanliness | Small disconnected component count | 25% |
| Proportions | Bounding box aspect ratio check | 10% |

### Segmentation Pipeline

```
Canvas PNG
    │
    ▼
Greyscale → Adaptive Threshold → Morphological Close
    │
    ▼
findContours (RETR_EXTERNAL)
    │
    ▼
Filter noise (area < 100 px²)
    │
    ▼
Iterative box merge (gap ≤ 15 px)  ← handles multi-stroke chars like '='
    │
    ▼
Sort left-to-right → extract 28×28 MNIST-normalised patches
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push the full repo to GitHub — including `.python-version` to pin Python 3.11
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Click **Deploy** — Streamlit Cloud handles the install and first-run model training

> **Tip:** Commit a pre-trained `smartboard_model.h5` to avoid the cold-start training delay on each deploy. The file is ~2 MB.

---

## 🌟 ML Portfolio Highlights

- **End-to-end ML pipeline** — data loading, preprocessing, training, evaluation, and live inference in one codebase
- **Production patterns** — `@st.cache_resource` model caching, graceful low-confidence fallbacks, full error handling
- **Computer vision** — adaptive thresholding, contour detection, morphological operations, distance transforms
- **Symbolic AI** — SymPy for exact algebraic solving with LaTeX output, not just numeric approximation
- **Interpretable ML** — quality score decomposed into 5 explainable CV metrics with actionable student tips
- **Full-stack deployment** — Streamlit + Streamlit Cloud, zero additional infrastructure

---

## 🔭 3 Extensions to Make It Stand Out

### 1. 🎨 Data Augmentation + Custom Symbol Training

Train a 17-class model (10 digits + `+ - × ÷ = ( )`) using synthetic operator images with `albumentations`:

```python
# augment.py
from albumentations import Compose, Rotate, ElasticTransform, GaussNoise
pipeline = Compose([Rotate(limit=15), ElasticTransform(alpha=1, sigma=50), GaussNoise()])
```

Full equation recognition end-to-end — no manual symbol mapping.

### 2. 📷 Real-time Webcam Mode

Add a webcam tab via `streamlit-webrtc` to recognise digits drawn on paper in real time:
- Frame capture → centre ROI crop → greyscale → threshold → CNN inference
- 5-frame running average to smooth predictions
- Makes for a compelling AR-style portfolio demo video

### 3. 🧮 Calculus & Trigonometry Solver

Extend `solver.py` with SymPy's `diff`, `integrate`, and `trigsimp`:

```python
import sympy as sp
x = sp.Symbol('x')

sp.diff(sp.sin(x)**2 + x**3, x)  # → 2·sin(x)·cos(x) + 3x²
sp.integrate(x**2, x)             # → x³/3
```

Pair with inline `matplotlib` plots — turns SmartBoard AI into a full symbolic math tutor.

---

## 📄 License

MIT — free to use, modify, and deploy.
