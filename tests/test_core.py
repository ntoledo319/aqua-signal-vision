import cv2
import numpy as np
import pytest

from opencv26 import core, synth


def test_clarity_clear_vs_bloom():
    ref = core._hue_hist(synth.make_clear_water(seed=99))
    clear = core.clarity_color_index(synth.make_clear_water(), ref)["clarity_color_index"]
    bloom = core.clarity_color_index(synth.make_bloom(), ref)["clarity_color_index"]
    assert clear < 0.35
    assert bloom > clear


def test_bloom_score_orders():
    clear = core.bloom_score(synth.make_clear_water())["bloom_score"]
    bloom = core.bloom_score(synth.make_bloom())["bloom_score"]
    assert bloom > 0.5
    assert clear < 0.15


def test_sheen_vs_ripple():
    sheen = core.surface_anomaly(synth.make_sheen())
    ripple = core.surface_anomaly(synth.make_ripple())
    assert sheen["surface_film_suspected"] is True
    assert ripple["surface_film_suspected"] is False
    assert ripple["laplacian_variance"] > sheen["laplacian_variance"]


def test_analyze_image_keys(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), synth.make_bloom())
    rep = core.analyze_image(str(p))
    for k in ("clarity_color_index", "bloom_score", "edge_density", "surface_film_suspected"):
        assert k in rep


def test_unreadable_image_raises():
    with pytest.raises(ValueError):
        core.load_bgr("/nonexistent/nope.png")


def test_trend_alert_on_rising_bloom(tmp_path):
    paths = []
    for i, green in enumerate([20, 60, 100, 140]):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (30, green, 30)
        p = tmp_path / f"f{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    trend = core.analyze_frames(paths)
    assert trend["frames"] == 4
    assert trend["bloom_slope"] > 0
    assert trend["alert"] is True


def test_trend_calm_water_no_alert(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"c{i}.png"
        cv2.imwrite(str(p), synth.make_clear_water(seed=100 + i))
        paths.append(str(p))
    trend = core.analyze_frames(paths)
    assert trend["alert"] is False


def test_video_cli(tmp_path):
    import subprocess, sys, json
    import cv2, numpy as np
    # build a tiny test video: clear -> bloom progression
    vw = cv2.VideoWriter(str(tmp_path / "t.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(20):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (100, 60 + i * 5, 60)
        vw.write(img)
    vw.release()
    r = subprocess.run([sys.executable, "-m", "opencv26", "video", str(tmp_path / "t.mp4"), "6"],
                       capture_output=True, text=True)
    assert r.returncode in (0, 2)
    out = json.loads(r.stdout)
    assert out["frames_sampled"] >= 4
    assert "bloom_slope" in out
