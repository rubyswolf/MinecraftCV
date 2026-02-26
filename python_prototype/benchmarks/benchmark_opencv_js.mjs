import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

function createRng(seed = 12345) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s;
  };
}

async function loadCv(opencvJsPath) {
  const abs = path.resolve(opencvJsPath);
  const mod = await import(pathToFileURL(abs).href);
  const candidate = mod.default ?? mod;
  if (typeof candidate === 'function') {
    return await candidate();
  }
  if (candidate && typeof candidate.then === 'function') {
    return await candidate;
  }
  return candidate;
}

function fillMatRandom(mat, seed = 12345) {
  const next = createRng(seed);
  const data = mat.data;
  for (let i = 0; i < data.length; i++) {
    data[i] = next() & 0xff;
  }
}

function benchCvtColor(cv, iterations = 300, warmup = 30) {
  const src = new cv.Mat(720, 1280, cv.CV_8UC3);
  const dst = new cv.Mat();
  fillMatRandom(src, 1001);

  for (let i = 0; i < warmup; i++) {
    cv.cvtColor(src, dst, cv.COLOR_BGR2GRAY);
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    cv.cvtColor(src, dst, cv.COLOR_BGR2GRAY);
  }
  const dt = performance.now() - t0;

  src.delete();
  dst.delete();
  return dt / iterations;
}

function benchLsd(cv, iterations = 120, warmup = 20) {
  const src = new cv.Mat(720, 1280, cv.CV_8UC1);
  const lines = new cv.Mat();
  fillMatRandom(src, 2002);

  const lsd = cv.createLineSegmentDetector();

  for (let i = 0; i < warmup; i++) {
    lsd.detect(src, lines);
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    lsd.detect(src, lines);
  }
  const dt = performance.now() - t0;

  lsd.delete();
  src.delete();
  lines.delete();
  return dt / iterations;
}

function benchSolvePnP(cv, iterations = 8000, warmup = 500) {
  const objectData = [
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 1.0, 0.0,
    1.0, 0.0, 1.0,
    0.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    2.0, 1.0, 0.0,
    0.0, 2.0, 1.0,
  ];
  const imageData = [
    639.4, 254.2,
    702.8, 290.6,
    603.2, 319.0,
    646.1, 196.5,
    666.0, 354.1,
    709.2, 233.0,
    609.4, 266.0,
    672.8, 302.5,
    735.0, 325.0,
    618.0, 201.0,
  ];

  const objectPoints = cv.matFromArray(10, 1, cv.CV_32FC3, objectData);
  const imagePoints = cv.matFromArray(10, 1, cv.CV_32FC2, imageData);
  const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
    1200.0, 0.0, 640.0,
    0.0, 1200.0, 360.0,
    0.0, 0.0, 1.0,
  ]);
  const distCoeffs = cv.matFromArray(4, 1, cv.CV_64FC1, [0.0, 0.0, 0.0, 0.0]);

  const rvec = new cv.Mat();
  const tvec = new cv.Mat();

  for (let i = 0; i < warmup; i++) {
    cv.solvePnP(
      objectPoints,
      imagePoints,
      cameraMatrix,
      distCoeffs,
      rvec,
      tvec,
      false,
      cv.SOLVEPNP_ITERATIVE
    );
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    cv.solvePnP(
      objectPoints,
      imagePoints,
      cameraMatrix,
      distCoeffs,
      rvec,
      tvec,
      false,
      cv.SOLVEPNP_ITERATIVE
    );
  }
  const dt = performance.now() - t0;

  objectPoints.delete();
  imagePoints.delete();
  cameraMatrix.delete();
  distCoeffs.delete();
  rvec.delete();
  tvec.delete();

  return dt / iterations;
}

async function main() {
  const opencvJsPath = process.argv[2] ?? 'build/opencv_js_mcv_single/bin/opencv.js';
  const cv = await loadCv(opencvJsPath);

  const results = {
    runtime: 'opencv.js',
    tasks_ms_per_op: {
      cvtColor_720p: benchCvtColor(cv),
      lsd_detect_720p: benchLsd(cv),
      solvePnP_10pts: benchSolvePnP(cv),
    },
  };

  console.log(JSON.stringify(results));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
