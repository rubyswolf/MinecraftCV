from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def clamp_rect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    xa = max(0, min(x0, x1))
    xb = min(width - 1, max(x0, x1))
    ya = max(0, min(y0, y1))
    yb = min(height - 1, max(y0, y1))

    if (xb - xa + 1) < min_size or (yb - ya + 1) < min_size:
        return None

    # Right/bottom are exclusive for slicing.
    return xa, ya, xb + 1, yb + 1


def detect_strongest_corners_in_roi(
    gray: np.ndarray,
    rect: tuple[int, int, int, int],
    max_corners: int,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max(1, max_corners),
        qualityLevel=0.01,
        minDistance=6,
        blockSize=3,
        useHarrisDetector=False,
    )

    if corners is None:
        return np.empty((0, 2), dtype=np.float32)

    corners_local = corners.reshape(-1, 2).astype(np.float32)

    # Subpixel refinement inside the selected ROI.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    corners_refine = corners_local.reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(roi, corners_refine, (5, 5), (-1, -1), criteria)

    corners_global = corners_refine.reshape(-1, 2)
    corners_global[:, 0] += float(x0)
    corners_global[:, 1] += float(y0)
    return corners_global


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive corner picker: drag a box and show strongest corners in that ROI."
    )
    parser.add_argument("--image", type=Path, default=Path("frame.png"))
    parser.add_argument("--max-corners", type=int, default=1)
    parser.add_argument("--min-box-size", type=int, default=12)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    image_path = args.image if args.image.is_absolute() else (base / args.image)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    state: dict[str, object] = {
        "drag_start": None,
        "drag_now": None,
        "selected_rect": None,
        "corners": np.empty((0, 2), dtype=np.float32),
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
            state["drag_now"] = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if state["drag_start"] is not None:
                state["drag_now"] = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if state["drag_start"] is None:
                return

            x0, y0 = state["drag_start"]  # type: ignore[index]
            rect = clamp_rect(int(x0), int(y0), int(x), int(y), w, h, max(4, int(args.min_box_size)))

            state["drag_start"] = None
            state["drag_now"] = None

            if rect is None:
                state["selected_rect"] = None
                state["corners"] = np.empty((0, 2), dtype=np.float32)
                return

            state["selected_rect"] = rect
            corners = detect_strongest_corners_in_roi(gray, rect, max_corners=max(1, int(args.max_corners)))
            state["corners"] = corners

            print(f"Selected ROI: {rect}, corners: {len(corners)}")
            for i, (cx, cy) in enumerate(corners, start=1):
                print(f"Corner {i}: ({cx:.3f}, {cy:.3f})")

    window = "cornerDetect"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = image.copy()

        selected_rect = state["selected_rect"]
        if isinstance(selected_rect, tuple):
            x0, y0, x1, y1 = selected_rect
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2, cv2.LINE_AA)

        drag_start = state["drag_start"]
        drag_now = state["drag_now"]
        if isinstance(drag_start, tuple) and isinstance(drag_now, tuple):
            cv2.rectangle(canvas, drag_start, drag_now, (255, 255, 0), 1, cv2.LINE_AA)

        corners = state["corners"]
        if isinstance(corners, np.ndarray):
            for cx, cy in corners:
                cv2.circle(canvas, (int(round(float(cx))), int(round(float(cy)))), 3, (0, 0, 255), -1, cv2.LINE_AA)

        help_text = "Drag LMB to select ROI and auto-detect strongest 10 corners | Esc=clear | q=quit"
        cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (20, 20, 20), 1, cv2.LINE_AA)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        if key == 27:  # Esc
            state["drag_start"] = None
            state["drag_now"] = None
            state["selected_rect"] = None
            state["corners"] = np.empty((0, 2), dtype=np.float32)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
