"""
quality.py — Handwriting quality scorer (0–100) using CV heuristics.
No ML model needed; interpretable feature engineering gives human-readable tips.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class QualityResult:
    score: int                   # 0–100
    grade: str                   # A / B / C / D / F
    breakdown: dict[str, float]  # component → sub-score (each 0–100)
    tips: list[str]              # actionable feedback strings
    color: str                   # CSS hex colour for the score display


# ─── Grading thresholds ───────────────────────────────────────────────────────

GRADE_THRESHOLDS = [
    (90, "A", "#22c55e"),  # green
    (75, "B", "#84cc16"),  # lime
    (60, "C", "#eab308"),  # amber
    (45, "D", "#f97316"),  # orange
    (0,  "F", "#ef4444"),  # red
]


def _grade(score: int) -> tuple[str, str]:
    for threshold, letter, color in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter, color
    return "F", "#ef4444"


# ─── Individual metric computations ───────────────────────────────────────────

def _stroke_consistency(binary: np.ndarray) -> float:
    """
    Measure consistency of stroke width via distance transform variance.
    Low variance → consistent strokes → high score.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_dists = dist[binary > 0]
    if len(ink_dists) < 10:
        return 50.0
    # Coefficient of variation (lower = more consistent)
    cv_val = np.std(ink_dists) / (np.mean(ink_dists) + 1e-9)
    score = max(0.0, 100.0 - cv_val * 100.0)
    return min(100.0, score)


def _ink_coverage(binary: np.ndarray) -> float:
    """
    Check that ink isn't too sparse (too light) or overly dense (too messy).
    Optimal density ≈ 5–25 % of canvas.
    """
    total = binary.size
    ink   = int(np.count_nonzero(binary))
    ratio = ink / total

    if ratio < 0.01:   # almost blank
        return 10.0
    if ratio > 0.5:    # extremely dense / noisy
        return 30.0
    # Score peaks around 8–15 %
    optimal = 0.10
    diff = abs(ratio - optimal)
    score = max(0.0, 100.0 - diff * 400.0)
    return min(100.0, score)


def _line_straightness(gray: np.ndarray) -> float:
    """
    Use Hough lines as a proxy for how straight/aligned the writing is.
    More lines detected with high confidence → better alignment.
    (Intentionally lenient for digits which are naturally curved.)
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)
    if lines is None:
        return 60.0  # neutral when no strong lines detected
    n = len(lines)
    score = min(100.0, 50.0 + n * 5.0)
    return score


def _noise_level(binary: np.ndarray) -> float:
    """
    Count tiny disconnected components (noise blobs).
    Fewer small components → less noise → higher score.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    min_area = 20
    noise_count = sum(
        1 for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] < min_area
    )
    score = max(0.0, 100.0 - noise_count * 10.0)
    return min(100.0, score)


def _aspect_uniformity(binary: np.ndarray) -> float:
    """
    Check that the overall ink bounding box has a reasonable aspect ratio.
    Extremely wide or tall handwriting may indicate distorted characters.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 50.0
    _, _, w, h = cv2.boundingRect(coords)
    if h == 0:
        return 50.0
    ar = w / h
    # Ideal range for a single digit: 0.5–1.5; equation: 2–8
    if 0.4 <= ar <= 8.0:
        return 85.0
    return 50.0


# ─── Main scorer ──────────────────────────────────────────────────────────────

def score_handwriting(gray_img: np.ndarray) -> QualityResult:
    """
    Score a grayscale handwriting image.

    Parameters
    ----------
    gray_img : np.ndarray  (H×W, uint8, white background / dark ink)

    Returns
    -------
    QualityResult
    """
    # Binarize
    _, binary = cv2.threshold(gray_img, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    breakdown = {
        "Stroke consistency":  _stroke_consistency(binary),
        "Ink coverage":        _ink_coverage(binary),
        "Alignment":           _line_straightness(gray_img),
        "Cleanliness":         _noise_level(binary),
        "Proportions":         _aspect_uniformity(binary),
    }

    # Weighted average
    weights = {
        "Stroke consistency": 0.30,
        "Ink coverage":       0.20,
        "Alignment":          0.15,
        "Cleanliness":        0.25,
        "Proportions":        0.10,
    }
    total = sum(breakdown[k] * weights[k] for k in breakdown)
    score = int(round(total))

    grade, color = _grade(score)
    tips = _generate_tips(breakdown, score)

    return QualityResult(
        score=score,
        grade=grade,
        breakdown=breakdown,
        tips=tips,
        color=color,
    )


# ─── Tip generator ────────────────────────────────────────────────────────────

def _generate_tips(breakdown: dict[str, float], score: int) -> list[str]:
    tips: list[str] = []

    if breakdown["Stroke consistency"] < 60:
        tips.append("✏️ Try to keep your stroke width consistent — hold the pen at a steady angle.")
    if breakdown["Ink coverage"] < 40:
        tips.append("📝 Your writing looks too light. Press a little harder for clearer strokes.")
    if breakdown["Ink coverage"] > 80 and breakdown["Cleanliness"] < 50:
        tips.append("🧹 There seems to be extra noise or smudging. Try to write more cleanly.")
    if breakdown["Alignment"] < 50:
        tips.append("📏 Keep your digits aligned on an imaginary baseline for neater writing.")
    if breakdown["Cleanliness"] < 50:
        tips.append("✨ Reduce stray marks and pen noise — lift the pen fully between strokes.")
    if breakdown["Proportions"] < 60:
        tips.append("📐 Check your digit proportions — avoid stretching numbers too wide or tall.")
    if score >= 90:
        tips.append("🌟 Excellent handwriting! Very clear and consistent strokes.")
    elif score >= 75:
        tips.append("👍 Good handwriting. A little more consistency will push you to excellent.")
    elif not tips:
        tips.append("🎯 Practice writing each digit slowly and deliberately to build muscle memory.")

    return tips[:4]  # cap at 4 tips to avoid overwhelming the user
