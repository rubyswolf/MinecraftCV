"""
Minimal OpenCV.js whitelist for MCV web tooling.

This is consumed by OpenCV's JS build system via:
  --config <path-to-this-file>
"""

core = {
    "": [
        "split",
        "normalize",
    ],
    "Algorithm": [],
}

imgproc = {
    "": [
        "cvtColor",
        "resize",
        "line",
        "circle",
        "rectangle",
        "putText",
        "createLineSegmentDetector",
    ],
    # Keep methods needed for LSD object usage (detector.detect(...)).
    "LineSegmentDetector": [
        "detect",
        "drawSegments",
        "compareSegments",
    ],
}

calib3d = {
    "": [
        "Rodrigues",
        "solvePnP",
        "solvePnPRansac",
        "solvePnPRefineLM",
        "projectPoints",
    ],
}

