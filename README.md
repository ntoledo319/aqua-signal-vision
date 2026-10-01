# aqua-signal Vision

Visual water-quality monitoring with OpenCV 5 — point a camera at water, get numbers.

**Demo video (2:30):** https://youtu.be/qkYewzKa7eU

- `python -m opencv26 analyze photo.jpg` → clarity index, bloom score, surface-film flags (JSON)
- `python -m opencv26 trend ./frames/` → time-series trend + alert (exit code 2 on alert)
- `python -m opencv26 webcam` → live loop

Built for the OpenCV AI Competition 2026. See CONCEPT.md. Tests: `pytest opencv26/tests` (7/7).

Demo gallery in `demo/` (real + synthetic, see ATTRIBUTION.md). Measured on real photos: bloom photo → bloom_score 0.137, clear stream → 0.005, oil sheen → film flag on synthetic sheen, clean ripple correctly negative.
