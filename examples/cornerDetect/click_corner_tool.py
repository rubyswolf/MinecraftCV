from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class LineCandidate:
    a: np.ndarray
    b: np.ndarray
    length: float
    angle: float
    score: float


@dataclass
class PendingSelection:
    click: np.ndarray
    mode: int
    lines: List[LineCandidate]


def normalize_angle_degrees(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def angle_diff_degrees(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 180.0 - diff)


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-10:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def point_to_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.linalg.norm(ab))
    if denom < 1e-10:
        return float("inf")
    cross = abs(float(ab[0] * (point[1] - a[1]) - ab[1] * (point[0] - a[0])))
    return cross / denom


def intersect_two_lines(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray | None:
    matrix = np.array(
        [[a2[0] - a1[0], b1[0] - b2[0]], [a2[1] - a1[1], b1[1] - b2[1]]],
        dtype=np.float64,
    )
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-10:
        return None
    rhs = np.array([b1[0] - a1[0], b1[1] - a1[1]], dtype=np.float64)
    t, _ = np.linalg.solve(matrix, rhs)
    return a1.astype(np.float64) + (a2.astype(np.float64) - a1.astype(np.float64)) * float(t)


def least_squares_intersection(lines: List[LineCandidate]) -> np.ndarray | None:
    if len(lines) < 2:
        return None

    a_rows = []
    b_rows = []
    for line in lines:
        direction = line.b.astype(np.float64) - line.a.astype(np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10:
            continue
        direction /= norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        a_rows.append(normal)
        b_rows.append(float(np.dot(normal, line.a.astype(np.float64))))

    if len(a_rows) < 2:
        return None

    a_mat = np.array(a_rows, dtype=np.float64)
    b_vec = np.array(b_rows, dtype=np.float64)
    if np.linalg.matrix_rank(a_mat) < 2:
        return None

    solution, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    return solution.astype(np.float64)


def refine_subpixel(gray: np.ndarray, point: np.ndarray, max_shift: float = 20.0) -> np.ndarray:
    p = np.array([[float(point[0]), float(point[1])]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-3)
    cv2.cornerSubPix(gray, p, (9, 9), (-1, -1), criteria)
    refined = p[0].astype(np.float64)
    if float(np.linalg.norm(refined - point.astype(np.float64))) > max_shift:
        return point.astype(np.float64)
    return refined


def build_line_candidates(
    raw_lines: np.ndarray,
    click: np.ndarray,
    image_shape: tuple[int, int, int],
) -> List[LineCandidate]:
    h, w = image_shape[:2]
    max_dist = 0.28 * float(max(h, w))
    min_len = max(15.0, 0.05 * float(min(h, w)))

    candidates: List[LineCandidate] = []
    for raw in raw_lines[:, 0, :]:
        a = np.array([float(raw[0]), float(raw[1])], dtype=np.float64)
        b = np.array([float(raw[2]), float(raw[3])], dtype=np.float64)
        length = float(np.linalg.norm(b - a))
        if length < min_len:
            continue

        dist_seg = point_to_segment_distance(click, a, b)
        if dist_seg > max_dist:
            continue

        dist_line = point_to_line_distance(click, a, b)
        angle = normalize_angle_degrees(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

        score = length / (1.0 + 0.9 * dist_seg + 0.7 * dist_line)
        candidates.append(LineCandidate(a=a, b=b, length=length, angle=angle, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_diverse_lines(candidates: List[LineCandidate], count: int) -> List[LineCandidate]:
    if count <= 0:
        return []

    min_sep = 18.0 if count >= 3 else 12.0
    selected: List[LineCandidate] = []

    for cand in candidates:
        if not selected:
            selected.append(cand)
        else:
            if all(angle_diff_degrees(cand.angle, s.angle) >= min_sep for s in selected):
                selected.append(cand)
        if len(selected) == count:
            return selected

    for cand in candidates:
        if cand in selected:
            continue
        selected.append(cand)
        if len(selected) == count:
            break

    return selected


def estimate_corner_from_selection(selection: PendingSelection) -> np.ndarray | None:
    if len(selection.lines) < 2:
        return None

    if selection.mode == 2:
        return intersect_two_lines(
            selection.lines[0].a,
            selection.lines[0].b,
            selection.lines[1].a,
            selection.lines[1].b,
        )

    return least_squares_intersection(selection.lines)


def draw_ui(
    base: np.ndarray,
    pending: PendingSelection | None,
    confirmed_points: List[np.ndarray],
) -> np.ndarray:
    canvas = base.copy()

    for pt in confirmed_points:
        cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 255), -1, cv2.LINE_AA)

    if pending is not None:
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for i, line in enumerate(pending.lines):
            color = colors[i % len(colors)]
            a = (int(round(line.a[0])), int(round(line.a[1])))
            b = (int(round(line.b[0])), int(round(line.b[1])))
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)

        click_xy = (int(round(pending.click[0])), int(round(pending.click[1])))
        cv2.circle(canvas, click_xy, 4, (0, 255, 0), -1, cv2.LINE_AA)

    text = "LMB=3 lines | RMB=2 lines | Space=confirm+refine | Esc=cancel | q=quit"
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("OpenCV contrib is required (cv2.ximgproc missing).")

    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "frame.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=10, do_merge=True)

    state = {
        "pending": None,
        "confirmed": [],
    }

    def start_selection(click_x: int, click_y: int, mode: int) -> None:
        click = np.array([float(click_x), float(click_y)], dtype=np.float64)
        raw = fld.detect(gray)
        if raw is None or len(raw) == 0:
            state["pending"] = None
            return

        candidates = build_line_candidates(raw, click, image.shape)
        selected = select_diverse_lines(candidates, mode)
        if len(selected) < 2:
            state["pending"] = None
            return

        state["pending"] = PendingSelection(click=click, mode=mode, lines=selected)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            start_selection(x, y, mode=3)
        elif event == cv2.EVENT_RBUTTONDOWN:
            start_selection(x, y, mode=2)

    window = "click_corner_tool"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        pending = state["pending"]
        confirmed = state["confirmed"]
        frame = draw_ui(image, pending, confirmed)
        cv2.imshow(window, frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

        if key == 27:  # Esc
            state["pending"] = None
            continue

        if key == 32:  # Space
            if pending is None:
                continue

            corner = estimate_corner_from_selection(pending)
            if corner is None:
                print("Could not estimate corner from current line selection.")
                state["pending"] = None
                continue

            refined = refine_subpixel(gray, corner)
            print(f"Corner: ({refined[0]:.3f}, {refined[1]:.3f}) mode={pending.mode}")
            state["confirmed"].append(refined)
            state["pending"] = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
