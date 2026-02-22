from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cornerDetect import (
    detect_faces,
    estimate_corner_from_faces,
    refine_corner_with_contrib_lines,
    require_contrib,
)


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def contains(self, point: tuple[float, float]) -> bool:
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass(frozen=True)
class TestCase:
    name: str
    scene: str
    template: str
    expected_corner_box: BoundingBox
    expected_face_count: int = 3


def run_case(case: TestCase, base_dir: Path) -> tuple[bool, str]:
    scene_path = base_dir / case.scene
    template_path = base_dir / case.template

    scene_bgr = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    if scene_bgr is None:
        return False, f"failed to read scene image: {scene_path}"
    if template_gray is None:
        return False, f"failed to read template image: {template_path}"

    detections = detect_faces(template_gray, scene_bgr)

    if len(detections) != case.expected_face_count:
        return (
            False,
            (
                f"unexpected detection count: got {len(detections)}, "
                f"expected {case.expected_face_count}"
            ),
        )

    coarse_corner = estimate_corner_from_faces(detections)
    if coarse_corner is None:
        return False, "corner not found"

    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    refined_corner = refine_corner_with_contrib_lines(scene_gray, detections, coarse_corner)
    corner_xy = (float(refined_corner[0]), float(refined_corner[1]))

    if not case.expected_corner_box.contains(corner_xy):
        box = case.expected_corner_box
        return (
            False,
            (
                f"corner out of bounds: ({corner_xy[0]:.3f}, {corner_xy[1]:.3f}) "
                f"not in box ({box.x}, {box.y}, w={box.w}, h={box.h})"
            ),
        )

    return True, f"corner=({corner_xy[0]:.3f}, {corner_xy[1]:.3f}), detections={len(detections)}"


def main() -> int:
    require_contrib()
    base_dir = Path(__file__).resolve().parent

    test_cases = [
        TestCase(
            name="cube polished andesite corner",
            scene="cube.png",
            template="polished_andesite.png",
            expected_corner_box=BoundingBox(147, 132, 6, 6),
        ),
        TestCase(
            name="thumbnail polished andesite corner",
            scene="thumbnail.png",
            template="polished_andesite.png",
            expected_corner_box=BoundingBox(636, 251, 8, 7),
        ),
        TestCase(
            name="small cube polished andesite corner",
            scene="small_cube.png",
            template="polished_andesite.png",
            expected_corner_box=BoundingBox(121, 136, 11, 10),
        ),
    ]

    passed = 0
    failed = 0

    for case in test_cases:
        ok, message = run_case(case, base_dir)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.name}: {message}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed, {len(test_cases)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
