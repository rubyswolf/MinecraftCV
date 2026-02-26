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

function parseArgInt(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx < 0 || idx + 1 >= process.argv.length) return fallback;
  const v = Number.parseInt(process.argv[idx + 1], 10);
  return Number.isFinite(v) ? v : fallback;
}

function benchmark(cv, { width = 1280, height = 720, warmup = 100, iterations = 2000 } = {}) {
  const src = new cv.Mat(height, width, cv.CV_8UC3);
  const dst = new cv.Mat();
  fillMatRandom(src, 9001);

  for (let i = 0; i < warmup; i++) {
    cv.cvtColor(src, dst, cv.COLOR_RGB2GRAY);
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    cv.cvtColor(src, dst, cv.COLOR_RGB2GRAY);
  }
  const dt = performance.now() - t0;

  src.delete();
  dst.delete();

  return {
    runtime: 'opencv-js',
    task: 'cvtColor_rgb2gray_720p',
    ms_per_op: dt / iterations,
    width,
    height,
    iterations,
  };
}

async function main() {
  const opencvJsPath =
    process.argv.find((arg) => !arg.startsWith('--') && arg.endsWith('.js')) ??
    'build/opencv_js_mcv_single/bin/opencv.js';
  const width = parseArgInt('width', 1280);
  const height = parseArgInt('height', 720);
  const warmup = parseArgInt('warmup', 100);
  const iterations = parseArgInt('iterations', 2000);

  const cv = await loadCv(opencvJsPath);
  console.log(JSON.stringify(benchmark(cv, { width, height, warmup, iterations })));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
