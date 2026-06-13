"""
preprocess.py — Image preprocessing and multi-digit segmentation.
Pipeline: raw canvas PNG → binarize → denoise → segment contours → 28×28 patches
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import io
from dataclasses import dataclass


@dataclass
class Segment:
    """A single segmented character region with its processed patch."""
    bbox: tuple[int, int, int, int]   # x, y, w, h in original image coords
    patch: np.ndarray                  # 28×28 float32 array, ready for CNN
    original_crop: np.ndarray          # raw grayscale crop (for display)


# ─── Core preprocessing ───────────────────────────────────────────────────────

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGBA/RGB image to OpenCV BGR uint8 array."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/grayscale array to PIL Image."""
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def preprocess_for_mnist(gray_crop: np.ndarray, target: int = 28) -> np.ndarray:
    """
    Replicate MNIST preprocessing:
    1. Invert (white bg → black bg)
    2. Threshold
    3. Find tight bounding box around ink
    4. Pad to square with 10 % margin
    5. Resize to 28×28
    6. Normalize to [0, 1] float32
    """
    # Invert: canvas is white bg / dark ink → MNIST is black bg / white ink
    inverted = cv2.bitwise_not(gray_crop)

    # Binarize
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    # Crop to ink bounding box
    coords = cv2.findNonZero(binary)
    if coords is None:
        # Empty region — return blank
        return np.zeros((target, target), dtype=np.float32)

    x0, y0, w0, h0 = cv2.boundingRect(coords)
    cropped = binary[y0:y0 + h0, x0:x0 + w0]

    # Pad to square
    side = max(w0, h0)
    pad = side // 5         # ~20 % padding per side like MNIST
    canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    off_x = pad + (side - w0) // 2
    off_y = pad + (side - h0) // 2
    canvas[off_y:off_y + h0, off_x:off_x + w0] = cropped

    # Resize
    resized = cv2.resize(canvas, (target, target), interpolation=cv2.INTER_AREA)

    return resized.astype(np.float32) / 255.0


def binarize_full_image(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold + morphological cleanup on the full canvas."""
    # Light denoising
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold handles uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=5,
    )

    # Morphological closing to fill small holes in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


# ─── Segmentation ─────────────────────────────────────────────────────────────

def segment_characters(
    pil_img: Image.Image,
    min_area: int = 100,
    merge_gap: int = 15,
) -> list[Segment]:
    """
    Detect and return individual character segments sorted left-to-right.

    Steps:
    1. Convert to grayscale
    2. Binarize (adaptive threshold)
    3. Find external contours
    4. Filter tiny noise
    5. Merge overlapping / near-adjacent bounding boxes (handles multi-stroke chars)
    6. Sort by x-coordinate
    7. Extract 28×28 patches for each segment
    """
    gray = np.array(pil_img.convert("L"))
    binary = binarize_full_image(gray)

    # Find contours on binary (ink = white)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect bounding boxes, filter noise
    boxes: list[list[int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]

    if not boxes:
        return []

    # Merge boxes that are close horizontally (same character strokes, e.g. '=', ':')
    boxes = _merge_nearby_boxes(boxes, gap=merge_gap)

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    segments: list[Segment] = []
    for (x1, y1, x2, y2) in boxes:
        # Slight padding
        pad = 4
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(gray.shape[1], x2 + pad)
        y2p = min(gray.shape[0], y2 + pad)

        crop = gray[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        patch = preprocess_for_mnist(crop)
        seg = Segment(
            bbox=(x1p, y1p, x2p - x1p, y2p - y1p),
            patch=patch,
            original_crop=crop,
        )
        segments.append(seg)

    return segments


def _merge_nearby_boxes(
    boxes: list[list[int]],
    gap: int = 15,
) -> list[list[int]]:
    """
    Iteratively merge bounding boxes that overlap or are within `gap` pixels.
    Uses a union-find-like repeated pass until stable.
    """
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        used = [False] * len(boxes)
        for i, b1 in enumerate(boxes):
            if used[i]:
                continue
            curr = b1[:]
            for j, b2 in enumerate(boxes):
                if i == j or used[j]:
                    continue
                # Check horizontal proximity (with gap tolerance)
                h_overlap = curr[0] - gap <= b2[2] and b2[0] - gap <= curr[2]
                v_overlap = curr[1] - gap <= b2[3] and b2[1] - gap <= curr[3]
                if h_overlap and v_overlap:
                    curr = [
                        min(curr[0], b2[0]),
                        min(curr[1], b2[1]),
                        max(curr[2], b2[2]),
                        max(curr[3], b2[3]),
                    ]
                    used[j] = True
                    changed = True
            merged.append(curr)
            used[i] = True
        boxes = merged
    return boxes


# ─── Utility ──────────────────────────────────────────────────────────────────

def draw_bboxes(pil_img: Image.Image, segments: list[Segment]) -> Image.Image:
    """Draw bounding boxes and index labels on a copy of the image."""
    cv_img = pil_to_cv(pil_img)
    for i, seg in enumerate(segments):
        x, y, w, h = seg.bbox
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (37, 99, 235), 2)
        cv2.putText(cv_img, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (37, 99, 235), 1)
    return cv_to_pil(cv_img)


def bytes_to_pil(raw: bytes) -> Image.Image:
    """Convert raw image bytes (PNG/JPG) to PIL Image."""
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def get_image_stats(gray: np.ndarray) -> dict:
    """Return basic stats for the grayscale image (used by quality scorer)."""
    non_zero = gray[gray < 200]  # ink pixels (dark on white bg)
    if len(non_zero) == 0:
        return {"ink_density": 0.0, "mean_intensity": 0.0, "std_intensity": 0.0}
    return {
        "ink_density": len(non_zero) / gray.size,
        "mean_intensity": float(np.mean(non_zero)),
        "std_intensity": float(np.std(non_zero)),
    }
