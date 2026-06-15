"""
image_quality.py — Image Quality Validator
==========================================
Checks blur, brightness, resolution, and colour saturation
before passing an image to the CNN.
Minimum passing score: 40 / 100
"""

from PIL import Image
import numpy as np


def check_image_quality(image_path: str) -> dict:
    """
    Validates an image and returns a quality dict:
      { score: int, passed: bool, checks: { resolution, brightness, sharpness, saturation } }
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"score": 0, "passed": False, "checks": {"error": str(e)}}

    arr = np.array(img, dtype=np.float32)
    score = 100
    checks = {}

    # ── Resolution ────────────────────────────────────────────────────────────
    w, h = img.size
    checks["resolution"] = w >= 100 and h >= 100
    if not checks["resolution"]:
        score -= 40

    # ── Brightness (mean pixel value 0–255) ───────────────────────────────────
    mean_brightness = float(arr.mean())
    checks["brightness"] = 40 <= mean_brightness <= 220
    if not checks["brightness"]:
        score -= 25

    # ── Sharpness (Laplacian variance — no scipy needed) ─────────────────────
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Approximate Laplacian with a simple kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Manual 2-D convolution (small kernel, no external deps)
    pad = np.pad(gray, 1, mode="reflect")
    h_g, w_g = gray.shape
    lap = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * pad[i:i+h_g, j:j+w_g]
    sharpness = float(lap.var())
    checks["sharpness"] = sharpness >= 80
    if not checks["sharpness"]:
        score -= 25

    # ── Colour saturation ─────────────────────────────────────────────────────
    sat = float(arr.max(axis=2).mean() - arr.min(axis=2).mean())
    checks["saturation"] = sat >= 10
    if not checks["saturation"]:
        score -= 10

    score = max(score, 0)

    return {
        "score": score,
        "passed": score >= 40,
        "checks": {
            "resolution":  checks["resolution"],
            "brightness":  checks["brightness"],
            "sharpness":   checks["sharpness"],
            "saturation":  checks["saturation"],
        }
    }
