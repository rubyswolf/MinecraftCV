import cv2
import json
import numpy as np
import time
from pathlib import Path


def bench_cvt_color(rng, iterations=300, warmup=30):
    img = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    for _ in range(warmup):
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    for _ in range(iterations):
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dt = time.perf_counter() - t0
    return (dt * 1000.0) / iterations


def bench_lsd(rng, iterations=120, warmup=20):
    img = rng.integers(0, 256, size=(720, 1280), dtype=np.uint8)
    lsd = cv2.createLineSegmentDetector()
    for _ in range(warmup):
        lsd.detect(img)
    t0 = time.perf_counter()
    for _ in range(iterations):
        lsd.detect(img)
    dt = time.perf_counter() - t0
    return (dt * 1000.0) / iterations


def bench_solvepnp(iterations=8000, warmup=500):
    # 3D points on a cube corner neighborhood (minecraft-like block corners)
    object_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
    ], dtype=np.float32)

    fx = fy = 1200.0
    cx, cy = 640.0, 360.0
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Fixed synthetic image points
    image_points = np.array([
        [639.4, 254.2],
        [702.8, 290.6],
        [603.2, 319.0],
        [646.1, 196.5],
        [666.0, 354.1],
        [709.2, 233.0],
        [609.4, 266.0],
        [672.8, 302.5],
        [735.0, 325.0],
        [618.0, 201.0],
    ], dtype=np.float32)

    for _ in range(warmup):
        cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    t0 = time.perf_counter()
    for _ in range(iterations):
        cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    dt = time.perf_counter() - t0
    return (dt * 1000.0) / iterations


def main():
    rng = np.random.default_rng(12345)
    results = {
        "runtime": "python-cv2",
        "opencv_version": cv2.__version__,
        "tasks_ms_per_op": {
            "cvtColor_720p": bench_cvt_color(rng),
            "lsd_detect_720p": bench_lsd(rng),
            "solvePnP_10pts": bench_solvepnp(),
        },
    }
    print(json.dumps(results))


if __name__ == "__main__":
    main()
