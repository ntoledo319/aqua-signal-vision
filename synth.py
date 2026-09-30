"""Synthetic test imagery: clear water, algae bloom, oil sheen, ripples.

Deterministic (seeded) so tests are reproducible. Images are meant to exercise
the analyzer's thresholds, not to be photorealistic.
"""
import cv2
import numpy as np


def make_clear_water(seed: int = 7, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (140, 100, 60)  # BGR ~ teal-blue
    ripple = (rng.normal(0, 8, (size, size, 1))).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ripple, 0, 255).astype(np.uint8)
    return img


def make_bloom(seed: int = 11, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (30, 110, 30)  # BGR ~ strong green
    noise = rng.normal(0, 10, (size, size, 1)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_sheen(seed: int = 3, size: int = 256) -> np.ndarray:
    """Smooth bright patch with rainbow hue gradient = oil-sheen-like."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    xx = np.linspace(0, 179, size, dtype=np.uint8)
    hsv[..., 0] = np.tile(xx, (size, 1))          # hue sweeps across frame
    hsv[..., 1] = 90
    hsv[..., 2] = 235                              # bright, low texture
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def make_ripple(seed: int = 5, size: int = 256) -> np.ndarray:
    """High-frequency edges everywhere = clean ripple texture."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (size, size), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
