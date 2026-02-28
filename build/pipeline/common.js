const fs = require("node:fs/promises");
const path = require("node:path");
const esbuild = require("esbuild");

const ROOT_DIR = path.resolve(__dirname, "..", "..");
const SRC_DIR = path.join(ROOT_DIR, "src");
const DIST_DIR = path.join(ROOT_DIR, "dist");
const DIST_COMMON_DIR = path.join(DIST_DIR, "common");

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function buildCommonArtifacts() {
  const options = arguments[0] || {};
  const backendMode = options.backendMode === "web" ? "web" : "python";
  const opencvUrl =
    typeof options.opencvUrl === "string" && options.opencvUrl.trim()
      ? options.opencvUrl
      : "/opencv.js";
  const mediaApiUrl =
    typeof options.mediaApiUrl === "string"
      ? options.mediaApiUrl
      : "";
  const dataApiUrl =
    typeof options.dataApiUrl === "string"
      ? options.dataApiUrl
      : "";

  await ensureDir(DIST_COMMON_DIR);

  const appEntry = path.join(SRC_DIR, "app.ts");
  const htmlTemplatePath = path.join(SRC_DIR, "template.html");
  const appBundlePath = path.join(DIST_COMMON_DIR, "app.bundle.js");
  const inlineHtmlPath = path.join(DIST_COMMON_DIR, "index.inline.html");

  const result = await esbuild.build({
    entryPoints: [appEntry],
    bundle: true,
    minify: true,
    format: "iife",
    platform: "browser",
    target: ["es2020"],
    legalComments: "none",
    write: false,
    logLevel: "silent",
    define: {
      __MCV_BACKEND__: JSON.stringify(backendMode),
      __MCV_OPENCV_URL__: JSON.stringify(opencvUrl),
      __MCV_MEDIA_API_URL__: JSON.stringify(mediaApiUrl),
      __MCV_DATA_API_URL__: JSON.stringify(dataApiUrl),
    },
  });

  const appBundle = result.outputFiles[0].text;
  await fs.writeFile(appBundlePath, appBundle, "utf8");

  const template = await fs.readFile(htmlTemplatePath, "utf8");
  const bodyStyleBlock =
    backendMode === "web"
      ? [
          "      background: transparent;",
          "      min-height: 100vh;",
          "      display: grid;",
          "      place-items: center;",
          "      padding: 24px;",
        ].join("\n")
      : [
          "      background: radial-gradient(1200px 600px at 30% -10%, #233247, var(--bg));",
          "      min-height: 100vh;",
          "      display: grid;",
          "      place-items: center;",
          "      padding: 24px;",
        ].join("\n");
  const safeInlineBundle = appBundle.replace(/<\/script/gi, "<\\/script");
  const backendScripts =
    backendMode === "web"
      ? `<script src="${opencvUrl.replace(/"/g, "&quot;")}"></script>`
      : "";
  const inlineHtml = template
    .replace("__MCV_BODY_STYLE__", () => bodyStyleBlock)
    .replace("__BACKEND_SCRIPTS__", () => backendScripts)
    .replace("__APP_JS__", () => safeInlineBundle);
  await fs.writeFile(inlineHtmlPath, inlineHtml, "utf8");

  return {
    rootDir: ROOT_DIR,
    distDir: DIST_DIR,
    distCommonDir: DIST_COMMON_DIR,
    appBundlePath,
    inlineHtmlPath,
    appBundle,
    inlineHtml,
  };
}

function renderPythonStandaloneScript(inlineHtml, requirementsText) {
  const htmlAsPythonString = JSON.stringify(inlineHtml);
  const requirementsAsPythonString = JSON.stringify(requirementsText);
  return `#!/usr/bin/env python3
import sys
from pathlib import Path
import base64
import time
import math

REQUIREMENTS_FILENAME = "mcv-requirements.txt"
REQUIREMENTS_TEXT = ${requirementsAsPythonString}


def write_requirements_file():
    output_path = Path.cwd() / REQUIREMENTS_FILENAME
    try:
        output_path.write_text(REQUIREMENTS_TEXT, encoding="utf-8")
        print(f"Wrote requirements file: {output_path}")
    except Exception as exc:
        print(f"Failed to write {REQUIREMENTS_FILENAME}: {exc}")
    return output_path

try:
    from flask import Flask, Response, jsonify, request
except ImportError as exc:
    requirements_path = write_requirements_file()
    print(f"Missing dependency: {exc.name}")
    print("Install requirements with:")
    print(f"  {sys.executable} -m pip install -r {requirements_path}")
    raise SystemExit(1)

try:
    import cv2
    import numpy as np
except ImportError as exc:
    requirements_path = write_requirements_file()
    print(f"Missing dependency: {exc.name}")
    print("Install requirements with:")
    print(f"  {sys.executable} -m pip install -r {requirements_path}")
    raise SystemExit(1)

HTML_PAGE = ${htmlAsPythonString}

app = Flask(__name__)


def parse_positive_float(value, default_value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    if parsed < 0:
        return 0.0
    return parsed


def decode_image_data_url_to_bgr(image_data_url):
    if not isinstance(image_data_url, str) or "," not in image_data_url:
        raise ValueError("Invalid image_data_url")
    header, encoded = image_data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("image_data_url must be base64 encoded")
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid image_data_url base64 payload") from exc
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image data")
    return image_bgr


def encode_png_data_url(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode PNG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def run_pipeline(args):
    started_at = time.time()
    image_data_url = args.get("image_data_url")
    image_bgr = decode_image_data_url_to_bgr(image_data_url)
    gray = np.mean(image_bgr, axis=2).astype(np.uint8)
    if not hasattr(cv2, "createLineSegmentDetector"):
        raise ValueError("LineSegmentDetector is unavailable in this OpenCV build")
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detect_result = lsd.detect(gray)
    lines = detect_result[0] if isinstance(detect_result, tuple) else detect_result
    line_segments = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            line_segments.append(
                [
                    float(raw[0]),
                    float(raw[1]),
                    float(raw[2]),
                    float(raw[3]),
                ]
            )
    duration_ms = int((time.time() - started_at) * 1000)
    return {
        "grayscale_image_data_url": encode_png_data_url(gray),
        "line_segments": line_segments,
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "duration_ms": duration_ms,
    }


def _is_finite_number(value):
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _extract_pose_correspondences(lines, vertices):
    if not isinstance(lines, list) or not isinstance(vertices, list):
        raise ValueError("lines and vertices must be arrays")
    out_rows = []
    max_line_index = len(lines) - 1
    for vertex in vertices:
        if not isinstance(vertex, dict):
            continue
        vx = vertex.get("x")
        vy = vertex.get("y")
        vz = vertex.get("z")
        if not (_is_finite_number(vx) and _is_finite_number(vy) and _is_finite_number(vz)):
            continue
        endpoint_ids = set()
        from_refs = vertex.get("from")
        to_refs = vertex.get("to")
        if isinstance(from_refs, list):
            for line_index in from_refs:
                if isinstance(line_index, int) and 0 <= line_index <= max_line_index:
                    endpoint_ids.add(line_index * 2)
        if isinstance(to_refs, list):
            for line_index in to_refs:
                if isinstance(line_index, int) and 0 <= line_index <= max_line_index:
                    endpoint_ids.add(line_index * 2 + 1)
        if not endpoint_ids:
            continue
        sum_x = 0.0
        sum_y = 0.0
        count = 0
        for endpoint_id in endpoint_ids:
            line_index = endpoint_id // 2
            endpoint_key = "from" if endpoint_id % 2 == 0 else "to"
            line = lines[line_index] if 0 <= line_index < len(lines) else None
            if not isinstance(line, dict):
                continue
            endpoint = line.get(endpoint_key)
            if not isinstance(endpoint, dict):
                continue
            px = endpoint.get("x")
            py = endpoint.get("y")
            if not (_is_finite_number(px) and _is_finite_number(py)):
                continue
            sum_x += float(px)
            sum_y += float(py)
            count += 1
        if count <= 0:
            continue
        out_rows.append((sum_x / count, sum_y / count, float(vx), float(vy), float(vz)))
    if len(out_rows) < 4:
        raise ValueError("Need at least 4 valid vertex correspondences with known world coordinates")
    data = np.array(out_rows, dtype=np.float64)
    image_pts = data[:, 0:2].astype(np.float64)
    object_pts = data[:, 2:5].astype(np.float64)
    return image_pts, object_pts


def _camera_matrix_from_focal(focal, width, height):
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _solve_pose_for_focal(object_pts, image_pts, width, height, focal):
    k = _camera_matrix_from_focal(focal, width, height)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts,
        image_pts,
        k,
        dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=4.0,
        confidence=0.999,
        iterationsCount=800,
    )
    if not ok:
        return float("inf"), np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), np.array([], dtype=np.int32), float("inf")

    if inliers is None or len(inliers) < 4:
        inlier_idx = np.arange(object_pts.shape[0], dtype=np.int32)
    else:
        inlier_idx = inliers.reshape(-1).astype(np.int32)

    obj_in = object_pts[inlier_idx]
    img_in = image_pts[inlier_idx]
    ok_refine, rvec, tvec = cv2.solvePnP(
        obj_in,
        img_in,
        k,
        dist,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok_refine:
        return float("inf"), np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64), inlier_idx, float("inf")

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, k, dist, rvec, tvec)
        except cv2.error:
            pass

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, k, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - image_pts, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    delta = 5.0
    huber = np.where(err <= delta, 0.5 * (err**2), delta * (err - 0.5 * delta))
    outlier_penalty = (object_pts.shape[0] - len(inlier_idx)) * (delta**2)
    cost = float(np.mean(huber) + outlier_penalty)
    return cost, rvec, tvec, inlier_idx, rmse


def _golden_section_search(fn, lo, hi, iterations=48):
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    inv_phi = 1.0 / phi
    x1 = hi - (hi - lo) * inv_phi
    x2 = lo + (hi - lo) * inv_phi
    f1 = fn(x1)
    f2 = fn(x2)
    for _ in range(iterations):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - (hi - lo) * inv_phi
            f1 = fn(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + (hi - lo) * inv_phi
            f2 = fn(x2)
    return (x1, f1) if f1 < f2 else (x2, f2)


def _wrap_degrees(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _pose_to_camera_world_and_minecraft_angles(rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    cam_world = -(rmat.T @ tvec).reshape(3)
    forward_world = rmat.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(forward_world))
    if n > 1e-12:
        forward_world = forward_world / n
    pitch = float(np.degrees(np.arcsin(np.clip(-forward_world[1], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(-forward_world[0], forward_world[2])))
    yaw = _wrap_degrees(yaw)
    return cam_world, pitch, yaw


def run_pose_solve(args):
    width_raw = args.get("width")
    height_raw = args.get("height")
    if not isinstance(width_raw, (int, float)) or not isinstance(height_raw, (int, float)):
        raise ValueError("width and height are required")
    width = int(width_raw)
    height = int(height_raw)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    lines = args.get("lines")
    vertices = args.get("vertices")
    image_pts, object_pts = _extract_pose_correspondences(lines, vertices)

    initial_vfov_raw = args.get("initial_vfov_deg")
    initial_vfov_deg = (
        float(initial_vfov_raw)
        if isinstance(initial_vfov_raw, (int, float)) and np.isfinite(float(initial_vfov_raw))
        else 70.0
    )
    if not (1.0 < initial_vfov_deg < 179.0):
        initial_vfov_deg = 70.0
    f_init = (height * 0.5) / np.tan(np.deg2rad(initial_vfov_deg * 0.5))
    f_min = max(20.0, f_init * 0.30)
    f_max = f_init * 3.00

    cache = {}

    def eval_logf(logf):
        key = float(logf)
        if key not in cache:
            focal = float(np.exp(logf))
            cache[key] = _solve_pose_for_focal(object_pts, image_pts, width, height, focal)
        return cache[key][0]

    logs = np.linspace(np.log(f_min), np.log(f_max), 44)
    costs = np.array([eval_logf(float(l)) for l in logs], dtype=np.float64)
    best_idx = int(np.argmin(costs))
    lo_idx = max(0, best_idx - 2)
    hi_idx = min(len(logs) - 1, best_idx + 2)
    lo = float(logs[lo_idx])
    hi = float(logs[hi_idx])
    if hi <= lo:
        lo = float(logs[0])
        hi = float(logs[-1])

    best_logf, _ = _golden_section_search(eval_logf, lo, hi, iterations=32)
    best_f = float(np.exp(best_logf))
    best_cost, best_rvec, best_tvec, best_inliers, rmse = _solve_pose_for_focal(
        object_pts, image_pts, width, height, best_f
    )
    if not np.isfinite(best_cost):
        raise RuntimeError("Focal search failed to find a valid PnP solution")

    cam_world, pitch_deg, yaw_deg = _pose_to_camera_world_and_minecraft_angles(best_rvec, best_tvec)
    player_y = float(cam_world[1] - 1.62)
    hfov_deg = float(np.degrees(2.0 * np.arctan((width * 0.5) / best_f)))
    vfov_deg = float(np.degrees(2.0 * np.arctan((height * 0.5) / best_f)))
    tp_command = (
        f"/tp @s {cam_world[0]:.6f} {player_y:.6f} {cam_world[2]:.6f} "
        f"{yaw_deg:.6f} {pitch_deg:.6f}"
    )

    return {
        "point_count": int(len(object_pts)),
        "inlier_count": int(len(best_inliers)),
        "image_width": int(width),
        "image_height": int(height),
        "initial_vfov_deg": float(initial_vfov_deg),
        "initial_focal_px": float(f_init),
        "optimized_focal_px": float(best_f),
        "optimized_hfov_deg": float(hfov_deg),
        "optimized_vfov_deg": float(vfov_deg),
        "reprojection_rmse_px": float(rmse),
        "camera_position": {
            "x": float(cam_world[0]),
            "y": float(cam_world[1]),
            "z": float(cam_world[2]),
        },
        "player_position": {
            "x": float(cam_world[0]),
            "y": float(player_y),
            "z": float(cam_world[2]),
        },
        "rotation": {
            "yaw": float(yaw_deg),
            "pitch": float(pitch_deg),
        },
        "tp_command": tp_command,
    }


def handle_mcv_cv_opencv_test(_args):
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "opencv_version": cv2.__version__,
        "gray_values": gray.reshape(-1).tolist(),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
        "mean_gray": float(np.mean(gray)),
    }


@app.get("/api/mcv/health")
def api_mcv_health():
    return jsonify({
        "ok": True,
        "backend": "python-cv2",
        "opencv_version": cv2.__version__,
    })


@app.post("/api/mcv")
def api_mcv():
    payload = request.get_json(silent=True) or {}
    op = payload.get("op")
    args = payload.get("args") or {}

    if op == "cv.opencvTest":
        return jsonify({"ok": True, "data": handle_mcv_cv_opencv_test(args)})
    if op == "cv.poseSolve":
        try:
            return jsonify({"ok": True, "data": run_pose_solve(args)})
        except Exception as exc:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": {
                            "code": "POSE_SOLVE_ERROR",
                            "message": str(exc),
                        },
                    }
                ),
                400,
            )

    return (
        jsonify(
            {
                "ok": False,
                "error": {
                    "code": "UNKNOWN_OP",
                    "message": f"Unsupported operation: {op}",
                },
            }
        ),
        400,
    )


@app.post("/api/mcv/pipeline")
def api_mcv_pipeline():
    payload = request.get_json(silent=True) or {}
    args = payload.get("args") or {}
    image_data_url = args.get("image_data_url")
    if not isinstance(image_data_url, str) or not image_data_url.strip():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "INVALID_ARGS",
                        "message": "image_data_url is required",
                    },
                }
            ),
            400,
        )

    try:
        result = run_pipeline(args)
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "PIPELINE_ERROR",
                        "message": str(exc),
                    },
                }
            ),
            400,
        )

    return jsonify({"ok": True, "data": result})


@app.get("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


def parse_port(argv):
    if len(argv) < 2:
        return 8765
    try:
        return int(argv[1])
    except ValueError:
        return 8765


if __name__ == "__main__":
    host = "127.0.0.1"
    port = parse_port(sys.argv)
    url = f"http://{host}:{port}/"
    print("MinecraftCV Python standalone server is running.")
    print("Open this link in your browser:")
    print(url)
    app.run(host=host, port=port, debug=False)
`;
}

module.exports = {
  ROOT_DIR,
  DIST_DIR,
  DIST_COMMON_DIR,
  ensureDir,
  buildCommonArtifacts,
  renderPythonStandaloneScript,
};
