from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_opencv_dir = repo_root / "build" / "opencv"
    default_output_dir = repo_root / "build" / "opencv_js_mcv_single"
    default_config = repo_root / "build" / "opencv_js_mcv_whitelist.py"

    parser = argparse.ArgumentParser(
        description="Build a single-file OpenCV.js bundle for MCV.",
    )
    parser.add_argument(
        "--opencv-dir",
        type=Path,
        default=default_opencv_dir,
        help="Path to OpenCV source (submodule root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output build directory for OpenCV.js artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Whitelist config file for OpenCV.js exports.",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
        help="CMake build type.",
    )
    parser.add_argument(
        "--simd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable WebAssembly SIMD (default: on).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output dir before configuring/building.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the OpenCV.js build. Without this flag, only print the command.",
    )
    return parser


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def validate_paths(opencv_dir: Path, config_path: Path) -> Path:
    build_js = opencv_dir / "platforms" / "js" / "build_js.py"
    if not build_js.exists():
        raise FileNotFoundError(f"OpenCV JS build script not found: {build_js}")
    if not config_path.exists():
        raise FileNotFoundError(f"Whitelist config not found: {config_path}")
    return build_js


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    opencv_dir = args.opencv_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    build_js = validate_paths(opencv_dir, config_path)

    cmd = [
        sys.executable,
        str(build_js),
        str(output_dir),
        "--opencv_dir",
        str(opencv_dir),
        "--build_wasm",
        "--config",
        str(config_path),
        f"--cmake_option=-DCMAKE_BUILD_TYPE={args.build_type}",
        "--cmake_option=-DBUILD_LIST=core,imgproc,calib3d",
    ]
    if args.simd:
        cmd.append("--simd")
    if args.clean:
        cmd.append("--clean_build_dir")

    print("Prepared OpenCV.js single-file build command:")
    print(quote_cmd(cmd))
    print()
    print("Notes:")
    print("- This command builds single-file output (no --disable_single_file).")
    print("- Run inside a shell where Emscripten is activated (emsdk_env).")
    print("- Expected output file: <output-dir>/bin/opencv.js")

    if not args.run:
        print()
        print("Dry run only. Re-run with --run to execute.")
        return

    env = os.environ.copy()
    if "EMSDK" not in env and "EMSCRIPTEN" not in env:
        print("Warning: EMSDK/EMSCRIPTEN env vars were not found. Build may fail if emcmake/emcc are not in PATH.")

    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()

