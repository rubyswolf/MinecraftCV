import json
import time
import argparse

import cv2
import numpy as np


def benchmark(width=1280, height=720, warmup=40, iterations=400):
    rng = np.random.default_rng(9001)
    src = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    for _ in range(warmup):
        cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    t0 = time.perf_counter()
    for _ in range(iterations):
        cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    dt = time.perf_counter() - t0

    return {
        "runtime": "python-cv2",
        "task": "cvtColor_rgb2gray_720p",
        "opencv_version": cv2.__version__,
        "ms_per_op": (dt * 1000.0) / iterations,
        "width": width,
        "height": height,
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    print(
        json.dumps(
            benchmark(
                width=args.width,
                height=args.height,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
    )


if __name__ == "__main__":
    main()
