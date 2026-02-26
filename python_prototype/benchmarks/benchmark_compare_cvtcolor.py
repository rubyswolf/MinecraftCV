import argparse
import json
import statistics
import subprocess
from pathlib import Path


def run_json(cmd, cwd):
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def median(values):
    return statistics.median(values)


def fmt(v):
    return f"{v:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python cvtColor vs OpenCV.js cvtColor vs manual JS grayscale."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--opencv-js-path",
        default="build/opencv_js_mcv_single/bin/opencv.js",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    py_cmd = [
        "python",
        "python_prototype/benchmarks/benchmark_python_cvtcolor_only.py",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    opencv_js_cmd = [
        "node",
        "python_prototype/benchmarks/benchmark_opencv_js_cvtcolor_only.mjs",
        args.opencv_js_path,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    manual_js_cmd = [
        "node",
        "python_prototype/benchmarks/benchmark_manual_gray_js.mjs",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]

    py_values = []
    opencv_js_values = []
    manual_js_values = []

    for _ in range(args.repeats):
        py_values.append(run_json(py_cmd, repo_root)["ms_per_op"])
        opencv_js_values.append(run_json(opencv_js_cmd, repo_root)["ms_per_op"])
        manual_js_values.append(run_json(manual_js_cmd, repo_root)["ms_per_op"])

    py_med = median(py_values)
    opencv_js_med = median(opencv_js_values)
    manual_js_med = median(manual_js_values)

    result = {
        "settings": {
            "width": args.width,
            "height": args.height,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "ms_per_op_median": {
            "python_cv2": py_med,
            "opencv_js_cvtColor": opencv_js_med,
            "manual_js_weighted": manual_js_med,
        },
        "speed_ratios": {
            "opencv_js_vs_python": opencv_js_med / py_med,
            "manual_js_vs_python": manual_js_med / py_med,
            "manual_js_vs_opencv_js": manual_js_med / opencv_js_med,
        },
        "raw_ms_per_op": {
            "python_cv2": py_values,
            "opencv_js_cvtColor": opencv_js_values,
            "manual_js_weighted": manual_js_values,
        },
    }

    print(json.dumps(result, indent=2))
    print()
    print("Summary:")
    print(f"- python cv2 median:      {fmt(py_med)} ms/op")
    print(f"- opencv.js median:       {fmt(opencv_js_med)} ms/op")
    print(f"- manual js median:       {fmt(manual_js_med)} ms/op")
    print(f"- opencv.js / python:     {fmt(opencv_js_med / py_med)}x")
    print(f"- manual js / python:     {fmt(manual_js_med / py_med)}x")
    print(f"- manual js / opencv.js:  {fmt(manual_js_med / opencv_js_med)}x")


if __name__ == "__main__":
    main()
