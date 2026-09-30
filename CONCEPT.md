# aqua-signal Vision — OpenCV AI Competition 2026 entry (Ledger)

**Competition:** OpenCV AI Competition 2026, powered by AWS — opencv26.devpost.com — $12K cash, due 2026-10-26 23:59 PT.
**Rule of note:** every entry must use OpenCV 5 for substantive image/video analysis and run as a real app. AWS compute grant not required for cash prizes.

## Concept
Freshwater visual monitor: point a camera/photo at a stream, lake, or tap sample and get quantitative visual water-quality indicators — no lab kit:
1. **Color/clarity index** — LAB/HSV histogram shift vs. reference clear-water palette (algae bloom greening, sediment browning).
2. **Surface debris / film detection** — texture + edge density + specular-reflection segmentation (oil sheen vs. clean ripple discrimination).
3. **Bloom scoring over time** — per-frame scores → trend alerts that feed the existing aqua-signal pipeline (USGS + narrative rules).
4. Optional live mode: webcam loop with rolling baseline.

## Why it wins (judging)
- Impact: citizen-science water monitoring, One Health framing (same story that already structures our OneAquaHealth entry).
- OpenCV 5 substantive: color-space analysis, segmentation, optical flow for ripple/sheen separation, feature tracking.
- Demo: runs on a laptop on bundled sample imagery + optional webcam; CLI + tiny web dashboard; tests included.

## Plan
- `opencv26/` package: capture, analyze, score, alert. Stdlib + opencv-python + numpy only.
- Sample data: bundled permissively-licensed stream/lake photos (document provenance) + synthetic generated references.
- Tests: pytest suite like glyphlab/aqua-signal (19/19, 43/43 bar).
- Demo video: 3-minute screen capture (we have the toolchain from the Era Wallet edit).

## Blocked-on
Devpost login (GitHub OAuth dead; password reset captcha). Build now, submit when restored.
