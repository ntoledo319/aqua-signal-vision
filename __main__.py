"""CLI: python -m opencv26 analyze IMG [IMG...] | trend DIR | webcam

Prints JSON reports. Exit code 2 when an alert fires (watchdog-friendly).
"""
import json
import sys
from pathlib import Path

from . import core


def _emit(report: dict) -> int:
    print(json.dumps(report, indent=2))
    return 2 if report.get("alert") or report.get("surface_film_suspected") else 0


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
