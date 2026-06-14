"""
image_quality.py — Image Quality Validator
==========================================
Checks uploaded lesion images for:
  - Brightness (too dark / overexposed)
  - Sharpness / blur (Laplacian variance)
  - Minimum resolution
  - Colour saturation (grey-only images)

Returns a score 0–100 and per-check details.
"""

import logging
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

MIN_DIMENSION = 100       # Minimum width/height in pixels
BLUR_THRESHOLD = 80       # Laplacian variance below this = blurry
BRIGHTNESS_MIN = 40       # 0–255 mean brightness — too dark
BRIGHTNESS_MAX = 220      # 0–255 mean brightness — overexposed
SATURATION_MIN = 10       # Low saturation may indicate greyscale/bad camera


def check_image_quality(image_path: str) -> dict:
    """
    Analyse image quality and return a score + per-check results.

    Returns:
        {
          "score": 0-100,
          "passed": bool,
          "checks": {
              "resolution": {"ok": bool, "detail": str},
              "brightness":  {"ok": bool, "detail": str},
              "sharpness":   {"ok": bool, "detail": str},
              "saturation":  {"ok": bool, "detail": str},
          }
        }
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Cannot open image for quality check: {e}")
        return {'score': 0, 'passed': False, 'checks': {}, 'error': str(e)}

    arr = np.array(img, dtype=np.float32)
    checks = {}
    deductions = 0

    # ── 1. Resolution ──────────────────────────────────────────────────────────
    w, h = img.size
    res_ok = w >= MIN_DIMENSION and h >= MIN_DIMENSION
    checks['resolution'] = {
        'ok': res_ok,
        'detail': f"{w}×{h}px" + ('' if res_ok else f' — minimum {MIN_DIMENSION}×{MIN_DIMENSION}px required')
    }
    if not res_ok:
        deductions += 40

    # ── 2. Brightness ──────────────────────────────────────────────────────────
    brightness = float(arr.mean())
    too_dark = brightness < BRIGHTNESS_MIN
    overexposed = brightness > BRIGHTNESS_MAX
    bright_ok = not too_dark and not overexposed
    checks['brightness'] = {
        'ok': bright_ok,
        'value': round(brightness, 1),
        'detail': (
            'Good lighting' if bright_ok
            else ('Image too dark — use better lighting' if too_dark else 'Image overexposed — reduce flash/light')
        )
    }
    if not bright_ok:
        deductions += 25

    # ── 3. Sharpness (Laplacian variance) ─────────────────────────────────────
    grey = img.convert('L')
    lap = grey.filter(ImageFilter.FIND_EDGES)
    lap_arr = np.array(lap, dtype=np.float32)
    sharpness = float(lap_arr.var())
    sharp_ok = sharpness >= BLUR_THRESHOLD
    checks['sharpness'] = {
        'ok': sharp_ok,
        'value': round(sharpness, 1),
        'detail': 'Sharp image' if sharp_ok else 'Image appears blurry — hold camera steady and focus on the lesion'
    }
    if not sharp_ok:
        deductions += 25

    # ── 4. Colour saturation ───────────────────────────────────────────────────
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    sat_ok = saturation >= SATURATION_MIN
    checks['saturation'] = {
        'ok': sat_ok,
        'value': round(saturation, 1),
        'detail': 'Good colour' if sat_ok else 'Low colour saturation — ensure the photo is in colour and well-lit'
    }
    if not sat_ok:
        deductions += 10

    score = max(0, 100 - deductions)
    return {
        'score': score,
        'passed': score >= 40,
        'checks': checks
    }
