"""aqua-signal Vision — OpenCV 5 water-quality visual analyzer.

Core analyses (all substantive OpenCV 5 work, no ML black boxes):
- clarity_color_index: LAB/HSV histogram shift vs. a clear-water reference palette
- surface_anomaly: oil-sheen vs clean-ripple discrimination via specular + texture
- bloom_score: green-channel dominance + hue concentration for algae estimation

All functions take/return plain numpy arrays and plain dicts. No global state.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def _hue_hist(bgr: np.ndarray, bins: int = 36) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 20), (179, 255, 255))  # ignore gray/shadow pixels
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def clarity_color_index(bgr: np.ndarray, reference_hist: np.ndarray | None = None) -> dict:
    """Histogram distance of the water's hue profile from a clear-water reference.

    reference_hist: L1-normalized hue histogram of clear water (36 bins). If None,
    a synthetic clear-water reference (blue-teal peak at H~100) is used.
    Returns 0.0 (matches clear reference) .. 1.0 (maximally shifted).
    """
    if reference_hist is None:
        reference_hist = np.zeros(36, dtype=np.float32)
        reference_hist[21] = 0.7   # OpenCV hue ~105-110: blue-teal clear water
        reference_hist[20] = 0.15
        reference_hist[22] = 0.15
    hist = _hue_hist(bgr, bins=len(reference_hist))
    dist = cv2.compareHist(reference_hist.astype(np.float32), hist.astype(np.float32),
                           cv2.HISTCMP_BHATTACHARYYA)
    return {"clarity_color_index": round(float(dist), 4)}


def bloom_score(bgr: np.ndarray) -> dict:
    """Algae-bloom proxy: share of pixels in the vegetation-green hue band,
    weighted by saturation. 0 = none visible, 1 = frame dominated by green bloom."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = cv2.inRange(hsv, (25, 60, 40), (90, 255, 255))
    sat_weight = s.astype(np.float32) / 255.0
    green_frac = float(np.count_nonzero(green)) / green.size
    green_sat = float((sat_weight * (green > 0)).mean())
    score = min(1.0, green_frac * 0.6 + green_sat * 0.4 * (green_frac > 0.02))
    return {"bloom_score": round(score, 4), "green_fraction": round(green_frac, 4)}


def surface_anomaly(bgr: np.ndarray) -> dict:
    """Detect surface films/sheens vs. clean ripple.

    Clean ripples: many short, high-frequency edges, low color coherence.
    Oil/foam films: large smooth specular regions with low texture (low Laplacian
    variance) and hue distortion (rainbow sheen => elevated hue stddev).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = float(lap.var())
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    bright = cv2.inRange(gray, 200, 255)
    specular_frac = float(np.count_nonzero(bright)) / bright.size
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_std = float(hsv[..., 0][bright > 0].std()) if np.count_nonzero(bright) else 0.0
    film_likely = specular_frac > 0.02 and texture < 150.0 and hue_std > 12.0
    return {
        "edge_density": round(edge_density, 4),
        "laplacian_variance": round(texture, 1),
        "specular_fraction": round(specular_frac, 4),
        "specular_hue_std": round(hue_std, 2),
        "surface_film_suspected": bool(film_likely),
    }


def analyze_image(path: str) -> dict:
    """Full single-image report."""
    bgr = load_bgr(path)
    out = {"image": path}
    out.update(clarity_color_index(bgr))
    out.update(bloom_score(bgr))
    out.update(surface_anomaly(bgr))
    return out


def analyze_frames(paths: list[str]) -> dict:
    """Trend over a time-ordered sequence of frames (for alerting)."""
    reports = [analyze_image(p) for p in paths]
    blooms = [r["bloom_score"] for r in reports]
    clarity = [r["clarity_color_index"] for r in reports]
    trend = {
        "frames": len(reports),
        "bloom_mean": round(float(np.mean(blooms)), 4),
        "bloom_slope": round(float(np.polyfit(range(len(blooms)), blooms, 1)[0]), 5) if len(blooms) > 1 else 0.0,
        "clarity_mean": round(float(np.mean(clarity)), 4),
        "clarity_slope": round(float(np.polyfit(range(len(clarity)), clarity, 1)[0]), 5) if len(clarity) > 1 else 0.0,
        "reports": reports,
    }
    trend["alert"] = trend["bloom_slope"] > 0.01 or trend["bloom_mean"] > 0.35
    return trend
