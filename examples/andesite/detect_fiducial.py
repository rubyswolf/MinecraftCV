from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def require_contrib() -> None:
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("OpenCV contrib modules are required (missing cv2.ximgproc).")


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (clockwise starting at top-left)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected exactly 4 points.")

    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    tl, tr = top_two[np.argsort(top_two[:, 0])]
    bl, br = bottom_two[np.argsort(bottom_two[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def point_to_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = point - a
    denom = float(np.linalg.norm(ab))
    if denom < 1e-8:
        return float("inf")
    cross = abs(float(ab[0] * ap[1] - ab[1] * ap[0]))
    return cross / denom


def line_intersection(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray | None:
    r = a2 - a1
    s = b2 - b1
    matrix = np.array([[r[0], -s[0]], [r[1], -s[1]]], dtype=np.float64)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-8:
        return None
    rhs = (b1 - a1).astype(np.float64)
    t, _ = np.linalg.solve(matrix, rhs)
    return a1 + (r * float(t))


def extract_cube_hexagon(scene_bgr: np.ndarray) -> np.ndarray:
    if scene_bgr.ndim != 3 or scene_bgr.shape[2] < 3:
        raise ValueError("Expected a BGR image.")

    fg_mask = (scene_bgr[:, :, :3].sum(axis=2) > 0).astype(np.uint8) * 255
    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No foreground contour found in cube image.")

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    for epsilon_scale in (0.005, 0.01, 0.015, 0.02, 0.03):
        approx = cv2.approxPolyDP(contour, epsilon_scale * perimeter, True)
        if len(approx) == 6:
            return approx.reshape(-1, 2).astype(np.float32)

    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    approx_hull = cv2.approxPolyDP(hull, 0.02 * hull_perimeter, True)
    if len(approx_hull) == 6:
        return approx_hull.reshape(-1, 2).astype(np.float32)

    raise RuntimeError("Could not extract a 6-point cube silhouette.")


def label_hexagon_vertices(hex_pts: np.ndarray) -> dict[str, np.ndarray]:
    pts = np.asarray(hex_pts, dtype=np.float32)
    if pts.shape != (6, 2):
        raise ValueError("Expected 6 silhouette vertices.")

    top_idx = int(np.argmin(pts[:, 1]))
    bottom_idx = int(np.argmax(pts[:, 1]))

    top = pts[top_idx]
    bottom = pts[bottom_idx]

    remaining_indices = [i for i in range(6) if i not in (top_idx, bottom_idx)]
    remaining = pts[remaining_indices]

    x_order = np.argsort(remaining[:, 0])
    left_two = remaining[x_order[:2]]
    right_two = remaining[x_order[2:]]

    left_upper = left_two[np.argmin(left_two[:, 1])]
    left_lower = left_two[np.argmax(left_two[:, 1])]
    right_upper = right_two[np.argmin(right_two[:, 1])]
    right_lower = right_two[np.argmax(right_two[:, 1])]

    return {
        "top": top,
        "left_upper": left_upper,
        "left_lower": left_lower,
        "bottom": bottom,
        "right_lower": right_lower,
        "right_upper": right_upper,
    }


def refine_front_top_with_contrib(
    scene_gray: np.ndarray,
    initial_front_top: np.ndarray,
) -> np.ndarray:
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=25, do_merge=True)
    lines = fld.detect(scene_gray)
    if lines is None or len(lines) == 0:
        return initial_front_top

    best_pos: tuple[float, tuple[np.ndarray, np.ndarray]] | None = None
    best_neg: tuple[float, tuple[np.ndarray, np.ndarray]] | None = None

    for raw_line in lines[:, 0, :]:
        a = np.array([raw_line[0], raw_line[1]], dtype=np.float32)
        b = np.array([raw_line[2], raw_line[3]], dtype=np.float32)
        length = float(np.linalg.norm(b - a))
        if length < 50.0:
            continue

        angle = math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))
        if angle < -90.0:
            angle += 180.0
        if angle > 90.0:
            angle -= 180.0

        distance = point_to_line_distance(initial_front_top, a, b)

        if 15.0 <= angle <= 45.0:
            if best_pos is None or distance < best_pos[0]:
                best_pos = (distance, (a, b))
        elif -45.0 <= angle <= -15.0:
            if best_neg is None or distance < best_neg[0]:
                best_neg = (distance, (a, b))

    if best_pos is None or best_neg is None:
        return initial_front_top

    intersect = line_intersection(best_pos[1][0], best_pos[1][1], best_neg[1][0], best_neg[1][1])
    if intersect is None:
        return initial_front_top

    if float(np.linalg.norm(intersect - initial_front_top)) <= 30.0:
        return intersect.astype(np.float32)
    return initial_front_top


def rotate_template_variants(template_gray: np.ndarray) -> List[np.ndarray]:
    return [
        template_gray,
        cv2.rotate(template_gray, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(template_gray, cv2.ROTATE_180),
        cv2.rotate(template_gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def face_match_score(face_patch: np.ndarray, template_variants: Iterable[np.ndarray]) -> float:
    best = -1.0
    face_eq = cv2.equalizeHist(face_patch)
    for variant in template_variants:
        raw = float(cv2.matchTemplate(face_patch, variant, cv2.TM_CCOEFF_NORMED)[0, 0])
        var_eq = cv2.equalizeHist(variant)
        eq = float(cv2.matchTemplate(face_eq, var_eq, cv2.TM_CCOEFF_NORMED)[0, 0])
        best = max(best, raw, eq)
    return best


def detect_faces(template_gray: np.ndarray, scene_bgr: np.ndarray) -> List[np.ndarray]:
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    hexagon = extract_cube_hexagon(scene_bgr)
    labels = label_hexagon_vertices(hexagon)

    vertical_left = labels["left_lower"] - labels["left_upper"]
    vertical_right = labels["right_lower"] - labels["right_upper"]
    vertical_avg = (vertical_left + vertical_right) * 0.5

    initial_front_top = labels["bottom"] - vertical_avg
    front_top = refine_front_top_with_contrib(scene_gray, initial_front_top)

    candidate_faces = [
        np.array([labels["left_upper"], labels["top"], labels["right_upper"], front_top], dtype=np.float32),
        np.array([labels["left_upper"], front_top, labels["bottom"], labels["left_lower"]], dtype=np.float32),
        np.array([front_top, labels["right_upper"], labels["right_lower"], labels["bottom"]], dtype=np.float32),
    ]

    h_t, w_t = template_gray.shape[:2]
    dst_square = np.array([[0, 0], [w_t - 1, 0], [w_t - 1, h_t - 1], [0, h_t - 1]], dtype=np.float32)
    template_variants = rotate_template_variants(template_gray)

    detections: List[np.ndarray] = []
    min_score = 0.30

    for quad in candidate_faces:
        transform = cv2.getPerspectiveTransform(quad, dst_square)
        patch = cv2.warpPerspective(scene_gray, transform, (w_t, h_t), flags=cv2.INTER_LINEAR)
        score = face_match_score(patch, template_variants)

        if score >= min_score:
            detections.append(order_points_clockwise(quad))

    return detections


def as_int_points(quad: np.ndarray) -> List[Point]:
    return [(int(round(x)), int(round(y))) for x, y in quad.tolist()]


def write_overlay(output_path: Path, scene_shape: tuple[int, int, int], detections: List[np.ndarray]) -> None:
    overlay = np.zeros((scene_shape[0], scene_shape[1], 4), dtype=np.uint8)
    rng = np.random.default_rng()

    for quad in detections:
        quad_int = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
        b, g, r = [int(v) for v in rng.integers(0, 256, size=3)]
        cv2.polylines(
            overlay,
            [quad_int],
            isClosed=True,
            color=(b, g, r, 220),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find repeated appearances of texture.png in cube.png and output transparent quadrilateral overlay."
        )
    )
    parser.add_argument("--template", type=Path, default=Path("texture.png"))
    parser.add_argument("--scene", type=Path, default=Path("cube.png"))
    parser.add_argument("--output", type=Path, default=Path("detection.png"))
    args = parser.parse_args()

    require_contrib()

    base = Path(__file__).resolve().parent
    template_path = args.template if args.template.is_absolute() else (base / args.template)
    scene_path = args.scene if args.scene.is_absolute() else (base / args.scene)
    output_path = args.output if args.output.is_absolute() else (base / args.output)

    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    scene_bgr = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)

    if template_gray is None:
        raise FileNotFoundError(f"Failed to load template image: {template_path}")
    if scene_bgr is None:
        raise FileNotFoundError(f"Failed to load scene image: {scene_path}")

    detections = detect_faces(template_gray, scene_bgr)

    if detections:
        for i, quad in enumerate(detections, start=1):
            print(f"Detection {i}: {as_int_points(quad)}")
    else:
        print("No detections found.")

    write_overlay(output_path, scene_bgr.shape, detections)


if __name__ == "__main__":
    main()
