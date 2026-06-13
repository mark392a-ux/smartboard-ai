"""
app.py — SmartBoard AI  |  Handwritten Math Learning Assistant
Run: streamlit run app.py
"""

from __future__ import annotations
import io
import time
import json
import base64
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go

# ── Local modules ─────────────────────────────────────────────────────────────
from model import load_or_train_model, predict_digit, get_top_k_predictions, DIGIT_LABELS
from preprocess import (
    pil_to_cv, segment_characters, draw_bboxes, bytes_to_pil, preprocess_for_mnist
)
from solver import solve_expression, labels_to_expression, is_valid_expression
from quality import score_handwriting

# ── Optional canvas (graceful fallback) ───────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

import cv2

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartBoard AI",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS — Swiss Minimalist + SaaS Educational ─────────────────────────
st.markdown("""
<style>
/* ── Import Inter ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --bg:        #F8FAFC;
    --surface:   #FFFFFF;
    --surface-2: #F1F5F9;
    --border:    #E2E8F0;
    --text-primary:   #0F172A;
    --text-secondary: #475569;
    --text-muted:     #94A3B8;
    --accent-blue:   #2563EB;
    --accent-teal:   #0D9488;
    --accent-green:  #16A34A;
    --accent-amber:  #D97706;
    --accent-red:    #DC2626;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1320px; }

/* ── Top nav bar ── */
.smartboard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.brand-name {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}
.brand-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 400;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #DCFCE7;
    color: #15803D;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #16A34A;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Card component ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.75rem;
}

/* ── Score ring ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.score-number {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.score-grade {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.conf-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    width: 20px;
    text-align: center;
    color: var(--text-primary);
}
.conf-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--surface-2);
    border-radius: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.conf-pct {
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 36px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ── Prediction result ── */
.prediction-hero {
    text-align: center;
    padding: 1rem 0;
}
.prediction-value {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-blue);
    line-height: 1;
}
.prediction-conf {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
}
.confidence-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.conf-high   { background: #DCFCE7; color: #15803D; }
.conf-medium { background: #FEF3C7; color: #92400E; }
.conf-low    { background: #FEE2E2; color: #991B1B; }

/* ── Solution card ── */
.solution-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
.step-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.step-number {
    min-width: 24px;
    height: 24px;
    background: var(--accent-blue);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-content {}
.step-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-bottom: 2px;
}
.step-explain {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ── Tip row ── */
.tip-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
}

/* ── History row ── */
.history-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
}
.history-expr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--accent-blue);
}
.history-result {
    font-size: 0.8rem;
    color: var(--text-secondary);
}
.history-time {
    margin-left: auto;
    font-size: 0.72rem;
    color: var(--text-muted);
}

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--surface-2);
    border-radius: var(--radius-md);
    padding: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-secondary);
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent-blue) !important;
    font-weight: 600;
    box-shadow: var(--shadow-sm);
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent-blue);
    color: white;
    border: none;
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 20px;
    transition: background 0.15s;
}
.stButton > button:hover {
    background: #1D4ED8;
}

/* ── Misc ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.empty-text { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "model": None,
        "history": [],
        "last_prediction": None,
        "last_quality": None,
        "last_solution": None,
        "last_segments": None,
        "stroke_width": 12,
        "drawing_mode": "freedraw",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_or_train_model()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="smartboard-header">
  <div class="brand">
    <div class="brand-icon">✏️</div>
    <div>
      <div class="brand-name">SmartBoard AI</div>
      <div class="brand-sub">Handwritten Math Learning Assistant</div>
    </div>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    Model Ready
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
with st.spinner("Loading SmartBoard AI model..."):
    model = get_model()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_draw, tab_upload, tab_examples, tab_history = st.tabs([
    "✏️  Draw", "📁  Upload", "💡  Examples", "📋  History"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DRAW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_draw:
    col_canvas, col_sidebar = st.columns([3, 2], gap="large")

    with col_canvas:
        # Controls row
        ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 1, 1])
        with ctrl_left:
            stroke_width = st.slider("Stroke width", 4, 28, 12, key="stroke_width_slider",
                                     label_visibility="visible")
        with ctrl_mid:
            stroke_color = st.color_picker("Ink color", "#1E293B", key="ink_color")
        with ctrl_right:
            st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
            clear_btn = st.button("🗑️ Clear canvas", key="clear_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        # Canvas
        st.markdown('<div class="card" style="padding:0; overflow:hidden;">', unsafe_allow_html=True)

        canvas_result = None
        if CANVAS_AVAILABLE:
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#FFFFFF",
                background_image=None,
                update_streamlit=True,
                height=340,
                width=680,
                drawing_mode="freedraw",
                display_toolbar=False,
                key="main_canvas",
            )
        else:
            st.warning("📦 Install `streamlit-drawable-canvas` for live drawing. "
                       "Upload an image in the **Upload** tab for now.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyse button
        analyse_col, _ = st.columns([1, 2])
        with analyse_col:
            analyse_btn = st.button("🔍 Analyse", key="analyse_draw", use_container_width=True)

    with col_sidebar:
        # ── Prediction card ───────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Live Prediction</div>', unsafe_allow_html=True)

        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            conf = pred["confidence"]
            conf_class = "conf-high" if conf >= 0.85 else ("conf-medium" if conf >= 0.5 else "conf-low")
            conf_label = "High confidence" if conf >= 0.85 else ("Uncertain" if conf >= 0.5 else "Low confidence")
            st.markdown(f"""
            <div class="prediction-hero">
                <div class="prediction-value">{pred['label']}</div>
                <div class="prediction-conf">
                    <span class="confidence-pill {conf_class}">{conf_label} &nbsp;{conf*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if pred.get("sequence"):
                st.markdown(f"""
                <div style="text-align:center; margin-top:-8px; margin-bottom:8px;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; 
                                 color:#0D9488; font-weight:600; letter-spacing:2px;">
                        {pred['sequence']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><div class="empty-icon">🎯</div>'
                        '<div class="empty-text">Draw something and click Analyse</div></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Confidence bars ───────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Confidence Distribution</div>', unsafe_allow_html=True)

        if st.session_state.last_prediction and "top_k" in st.session_state.last_prediction:
            top_k = st.session_state.last_prediction["top_k"]
            bars_html = ""
            colors = ["#2563EB", "#0D9488", "#7C3AED", "#D97706", "#DC2626"]
            for i, (lbl, p) in enumerate(top_k):
                col_hex = colors[i % len(colors)]
                pct = p * 100
                bars_html += f"""
                <div class="conf-row">
                    <div class="conf-label">{lbl}</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{col_hex};"></div>
                    </div>
                    <div class="conf-pct">{pct:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--text-muted); font-size:0.82rem; text-align:center; '
                        'padding:1rem 0;">No predictions yet</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Quality score ─────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Handwriting Quality</div>', unsafe_allow_html=True)

        if st.session_state.last_quality:
            q = st.session_state.last_quality
            st.markdown(f"""
            <div class="score-ring-wrap" style="margin-bottom:12px;">
                <div class="score-number" style="color:{q.color};">{q.score}</div>
                <div class="score-grade">Grade {q.grade} &nbsp;/&nbsp; 100</div>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown mini-bars
            for metric, val in q.breakdown.items():
                pct = val
                st.markdown(f"""
                <div class="conf-row">
                    <div style="font-size:0.72rem; color:var(--text-secondary); width:130px;">{metric}</div>
                    <div class="conf-bar-bg" style="flex:1;">
                        <div class="conf-bar-fill" style="width:{pct:.0f}%; background:{q.color};"></div>
                    </div>
                    <div class="conf-pct">{pct:.0f}</div>
                </div>""", unsafe_allow_html=True)

            if q.tips:
                st.markdown("<hr style='margin:10px 0; border-color:var(--border);'>", unsafe_allow_html=True)
                for tip in q.tips:
                    st.markdown(f'<div class="tip-row">{tip}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--text-muted); font-size:0.82rem; text-align:center; '
                        'padding:1rem 0;">Draw to get quality feedback</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Solution panel (full width below) ────────────────────────────────────
    if st.session_state.last_solution:
        sol = st.session_state.last_solution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        emoji = "🧮" if sol.is_equation else "✅"
        st.markdown(f'<div class="solution-header">{emoji} Step-by-Step Solution</div>',
                    unsafe_allow_html=True)

        sol_left, sol_right = st.columns([1, 1])
        with sol_left:
            for i, step in enumerate(sol.steps):
                st.markdown(f"""
                <div class="step-row">
                  <div class="step-number">{i+1}</div>
                  <div class="step-content">
                    <div class="step-label">{step['label']}</div>
                    <div class="step-explain">{step['explanation']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with sol_right:
            for step in sol.steps:
                try:
                    st.latex(step["latex"])
                except Exception:
                    st.code(step["latex"])

        if not sol.error:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #EFF6FF, #F0FDF4);
                        border: 1px solid #BFDBFE; border-radius: var(--radius-md);
                        padding: 14px 20px; display:flex; align-items:center; gap:10px; margin-top:8px;">
                <span style="font-size:1.4rem;">🎯</span>
                <div>
                    <div style="font-size:0.72rem; font-weight:600; color:#1D4ED8; 
                                text-transform:uppercase; letter-spacing:0.5px;">Answer</div>
                    <div style="font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;
                                color:#0F172A;">{sol.result}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ {sol.error}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Segmentation preview ──────────────────────────────────────────────────
    if st.session_state.last_segments:
        with st.expander("🔬 Segmentation preview (processed image)", expanded=False):
            seg_img, patches = st.session_state.last_segments
            st.image(seg_img, caption="Detected character regions", use_container_width=False, width=400)
            if patches:
                st.markdown("**28×28 patches fed to CNN:**")
                patch_cols = st.columns(min(len(patches), 10))
                for i, (patch, lbl) in enumerate(patches[:10]):
                    with patch_cols[i]:
                        fig, ax = plt.subplots(figsize=(1.5, 1.5))
                        ax.imshow(patch, cmap='gray', vmin=0, vmax=1)
                        ax.set_title(str(lbl), fontsize=10, pad=2)
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)


# ─── Handle Analyse button ────────────────────────────────────────────────────
def run_analysis(pil_img: Image.Image):
    """Core analysis pipeline — runs on draw or upload."""
    # 1. Segment
    segments = segment_characters(pil_img)

    if not segments:
        st.warning("No ink detected. Draw a digit or expression and click Analyse.")
        return

    # 2. Predict each segment
    labels, confs, all_probs_list = [], [], []
    for seg in segments:
        cls, conf, probs = predict_digit(model, seg.patch)
        labels.append(DIGIT_LABELS[cls])
        confs.append(conf)
        all_probs_list.append(probs)

    # 3. Build expression string
    sequence = labels_to_expression(labels)
    avg_conf = float(np.mean(confs))

    # Use first segment stats for single-digit display
    top_k = get_top_k_predictions(all_probs_list[0], k=5)

    st.session_state.last_prediction = {
        "label": labels[0] if len(labels) == 1 else sequence,
        "confidence": confs[0] if len(confs) == 1 else avg_conf,
        "sequence": sequence if len(labels) > 1 else None,
        "top_k": top_k,
    }

    # 4. Quality score — use full image
    gray = np.array(pil_img.convert("L"))
    quality = score_handwriting(gray)
    st.session_state.last_quality = quality

    # 5. Solve if valid expression
    if is_valid_expression(sequence):
        solution = solve_expression(sequence)
        st.session_state.last_solution = solution

        # Add to history
        st.session_state.history.insert(0, {
            "expression": sequence,
            "result": solution.result,
            "confidence": f"{avg_conf*100:.0f}%",
            "quality": quality.score,
            "time": datetime.now().strftime("%H:%M:%S"),
            "error": solution.error,
        })
        st.session_state.history = st.session_state.history[:50]  # cap
    else:
        st.session_state.last_solution = None
        if avg_conf < 0.5:
            st.info(f"Low confidence ({avg_conf*100:.0f}%). Try writing more clearly "
                    f"or increase stroke width.")

    # 6. Segmentation preview
    annotated = draw_bboxes(pil_img, segments)
    patch_pairs = [(seg.patch, lbl) for seg, lbl in zip(segments, labels)]
    st.session_state.last_segments = (annotated, patch_pairs)


if analyse_btn and CANVAS_AVAILABLE and canvas_result is not None:
    if canvas_result.image_data is not None:
        img_arr = canvas_result.image_data.astype(np.uint8)
        pil_img = Image.fromarray(img_arr, mode="RGBA").convert("RGB")
        # Check if canvas has any ink (non-white pixels)
        gray_check = np.array(pil_img.convert("L"))
        if gray_check.min() < 240:
            with st.spinner("Analysing..."):
                run_analysis(pil_img)
            st.rerun()
        else:
            st.warning("Canvas appears empty. Draw something first!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    up_left, up_right = st.columns([1, 1], gap="large")

    with up_left:
        st.markdown("### Upload a handwritten math image")
        st.markdown('<p style="color:var(--text-secondary); font-size:0.85rem;">'
                    'Supported: PNG, JPG, JPEG. Best results with dark ink on white background.</p>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            key="upload_file",
            label_visibility="collapsed"
        )

        if uploaded_file:
            img_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

            if st.button("🔍 Analyse Image", key="analyse_upload", use_container_width=True):
                with st.spinner("Analysing..."):
                    run_analysis(pil_img)
                st.rerun()

    with up_right:
        if st.session_state.last_prediction:
            st.markdown("### Results")
            pred = st.session_state.last_prediction
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Recognised</div>
                <div class="prediction-hero">
                    <div class="prediction-value" style="font-size:2.5rem;">{pred['label']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if st.session_state.last_solution:
                sol = st.session_state.last_solution
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Solution</div>', unsafe_allow_html=True)
                for step in sol.steps:
                    try:
                        st.latex(step["latex"])
                    except Exception:
                        st.code(step["latex"])
                if not sol.error:
                    st.success(f"**Answer: {sol.result}**")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><div class="empty-icon">📁</div>'
                        '<div class="empty-text">Upload an image and click Analyse</div></div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_examples:
    st.markdown("### Try these example expressions")
    st.markdown('<p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:1.5rem;">'
                'Click any expression to see the full step-by-step solution.</p>',
                unsafe_allow_html=True)

    examples = [
        {"expr": "3+4",       "desc": "Simple addition",         "category": "Arithmetic"},
        {"expr": "12-7",      "desc": "Subtraction",             "category": "Arithmetic"},
        {"expr": "6*8",       "desc": "Multiplication",          "category": "Arithmetic"},
        {"expr": "84/4",      "desc": "Division",                "category": "Arithmetic"},
        {"expr": "(3+4)*2",   "desc": "Brackets first",          "category": "Arithmetic"},
        {"expr": "2**8",      "desc": "Exponentiation",          "category": "Arithmetic"},
        {"expr": "2*x+6",     "desc": "Linear expression",       "category": "Algebra"},
        {"expr": "2*x+6=10",  "desc": "Linear equation",        "category": "Algebra"},
        {"expr": "x**2-5*x+6=0", "desc": "Quadratic equation",  "category": "Algebra"},
        {"expr": "3*x+2*y",   "desc": "Two-variable expression", "category": "Algebra"},
    ]

    categories = list(dict.fromkeys(e["category"] for e in examples))
    for cat in categories:
        st.markdown(f"**{cat}**")
        cat_examples = [e for e in examples if e["category"] == cat]
        cols = st.columns(len(cat_examples))
        for col, ex in zip(cols, cat_examples):
            with col:
                st.markdown(f"""
                <div class="card" style="padding:1rem; min-height:90px;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem;
                                font-weight:600; color:var(--accent-blue); margin-bottom:4px;">
                        {ex['expr']}
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted);">{ex['desc']}</div>
                </div>""", unsafe_allow_html=True)
                if st.button("Solve", key=f"ex_{ex['expr']}", use_container_width=True):
                    sol = solve_expression(ex["expr"])
                    st.session_state.last_solution = sol
                    st.session_state.last_prediction = {
                        "label": ex["expr"],
                        "confidence": 1.0,
                        "sequence": ex["expr"],
                        "top_k": [("—", 1.0)] + [("—", 0.0)] * 4,
                    }
                    st.rerun()

    if st.session_state.last_solution and st.session_state.last_prediction:
        sol = st.session_state.last_solution
        expr = st.session_state.last_prediction.get("label", "")
        st.markdown(f"---\n### Solution: `{expr}`")
        ex_cols = st.columns([1, 1])
        with ex_cols[0]:
            for i, step in enumerate(sol.steps):
                st.markdown(f"""
                <div class="step-row">
                  <div class="step-number">{i+1}</div>
                  <div class="step-content">
                    <div class="step-label">{step['label']}</div>
                    <div class="step-explain">{step['explanation']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
        with ex_cols[1]:
            for step in sol.steps:
                try:
                    st.latex(step["latex"])
                except Exception:
                    st.code(step["latex"])
        if not sol.error:
            st.success(f"✅ **Answer: {sol.result}**")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.markdown('<div class="empty-state"><div class="empty-icon">📋</div>'
                    '<div class="empty-text">Your solved expressions will appear here.</div></div>',
                    unsafe_allow_html=True)
    else:
        h_left, h_right = st.columns([2, 1])
        with h_left:
            st.markdown(f"### {len(st.session_state.history)} solved expressions")
            for item in st.session_state.history:
                status = "✅" if not item.get("error") else "⚠️"
                st.markdown(f"""
                <div class="history-row">
                    <div>{status}</div>
                    <div>
                        <div class="history-expr">{item['expression']}</div>
                        <div class="history-result">= {item['result']}</div>
                    </div>
                    <div style="margin-left:auto; text-align:right;">
                        <div style="font-size:0.72rem; color:var(--text-muted);">{item['time']}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted);">
                            Conf: {item['confidence']} · Quality: {item['quality']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with h_right:
            # Mini stats
            solved   = [h for h in st.session_state.history if not h.get("error")]
            avg_qual = np.mean([h["quality"] for h in st.session_state.history]) if st.session_state.history else 0
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Session Stats</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px;">
                    <div style="text-align:center;">
                        <div style="font-size:2rem; font-weight:700; color:var(--accent-blue);">
                            {len(st.session_state.history)}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted);">Expressions</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem; font-weight:700; color:var(--accent-teal);">
                            {avg_qual:.0f}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted);">Avg Quality</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem; font-weight:700; color:var(--accent-green);">
                            {len(solved)}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted);">Solved</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem; font-weight:700; color:var(--accent-amber);">
                            {len(st.session_state.history) - len(solved)}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted);">With Errors</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            if st.button("🗑️ Clear history", key="clear_history"):
                st.session_state.history = []
                st.rerun()

            # Confidence trend
            if len(st.session_state.history) >= 3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Quality Trend</div>', unsafe_allow_html=True)
                scores = [h["quality"] for h in reversed(st.session_state.history)]
                fig = go.Figure(go.Scatter(
                    x=list(range(1, len(scores) + 1)),
                    y=scores,
                    mode='lines+markers',
                    line=dict(color='#2563EB', width=2),
                    marker=dict(size=6, color='#2563EB'),
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=160,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title="Attempt"),
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9', range=[0, 100]),
                    font=dict(family='Inter', size=11),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:var(--text-muted); font-size:0.75rem;
            border-top:1px solid var(--border); margin-top:2rem;">
    SmartBoard AI &nbsp;·&nbsp; CNN · OpenCV · SymPy · Streamlit
    &nbsp;·&nbsp; <a href="https://github.com" style="color:var(--accent-blue); text-decoration:none;">
    GitHub</a>
</div>
""", unsafe_allow_html=True)
