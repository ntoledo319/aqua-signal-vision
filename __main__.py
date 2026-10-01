"""CLI: python -m opencv26 analyze IMG [IMG...] | trend DIR | video FILE [N] | webcam

Prints JSON reports. Exit code 2 when an alert fires (watchdog-friendly).
"""
import json
import sys
from pathlib import Path

from . import core


def _emit(report: dict) -> int:
    print(json.dumps(report, indent=2))
    return 2 if report.get("alert") or report.get("surface_film_suspected") else 0


def _video(rest: list[str]) -> int:
    """Sample N frames evenly from a video file and run trend analysis."""
    import cv2
    import numpy as np

    path = rest[0]
    n = int(rest[1]) if len(rest) > 1 else 12
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"unreadable video: {path}", file=sys.stderr)
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    reports = []
    for i in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rep = {"frame": int(i)}
        rep.update(core.clarity_color_index(frame))
        rep.update(core.bloom_score(frame))
        reports.append(rep)
    cap.release()
    if not reports:
        print("no frames decoded", file=sys.stderr)
        return 1
    blooms = [r["bloom_score"] for r in reports]
    out = {
        "video": path,
        "frames_sampled": len(reports),
        "bloom_mean": round(float(np.mean(blooms)), 4),
        "bloom_max": round(float(np.max(blooms)), 4),
        "bloom_slope": round(float(np.polyfit(range(len(blooms)), blooms, 1)[0]), 5) if len(blooms) > 1 else 0.0,
        "reports": reports,
    }
    out["alert"] = out["bloom_slope"] > 0.01 or out["bloom_mean"] > 0.35
    return _emit(out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "analyze" and rest:
        rc = 0
        for img in rest:
            rc = max(rc, _emit(core.analyze_image(img)))
        return rc
    if cmd == "trend" and rest:
        frames = sorted(str(p) for p in Path(rest[0]).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not frames:
            print("no frames found", file=sys.stderr)
            return 1
        return _emit(core.analyze_frames(frames))
    if cmd == "video" and rest:
        return _video(rest)
    if cmd == "webcam":
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("no webcam", file=sys.stderr)
            return 1
        print("press q to quit; JSON report every 30 frames", file=sys.stderr)
        n, window = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("aqua-signal vision", frame)
            if n % 30 == 0:
                rep = {}
                rep.update(core.clarity_color_index(frame))
                rep.update(core.bloom_score(frame))
                print(json.dumps(rep), file=sys.stderr)
            n += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
