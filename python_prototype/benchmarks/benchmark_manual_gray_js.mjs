import { performance } from 'node:perf_hooks';

function createRng(seed = 12345) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s;
  };
}

function makeRandomRgb(width, height, seed = 12345) {
  const src = new Uint8Array(width * height * 3);
  const next = createRng(seed);
  for (let i = 0; i < src.length; i++) {
    src[i] = next() & 0xff;
  }
  return src;
}

// Weighted grayscale approximation:
// gray ~= (0.299 * R + 0.587 * G + 0.114 * B)
// Integer form: (77*R + 150*G + 29*B + 128) >> 8
function rgbToGrayWeighted(srcRgb, dstGray) {
  let si = 0;
  let di = 0;
  const n = dstGray.length;
  while (di < n) {
    const r = srcRgb[si];
    const g = srcRgb[si + 1];
    const b = srcRgb[si + 2];
    dstGray[di] = (77 * r + 150 * g + 29 * b + 128) >> 8;
    si += 3;
    di += 1;
  }
}

function benchmark({
  width = 1280,
  height = 720,
  warmup = 40,
  iterations = 400,
} = {}) {
  const src = makeRandomRgb(width, height, 9001);
  const dst = new Uint8Array(width * height);

  for (let i = 0; i < warmup; i++) {
    rgbToGrayWeighted(src, dst);
  }

  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) {
    rgbToGrayWeighted(src, dst);
  }
  const dt = performance.now() - t0;

  return {
    runtime: 'manual-js',
    task: 'rgb_to_gray_weighted_720p',
    ms_per_op: dt / iterations,
    width,
    height,
    iterations,
  };
}

function parseArgInt(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx < 0 || idx + 1 >= process.argv.length) return fallback;
  const v = Number.parseInt(process.argv[idx + 1], 10);
  return Number.isFinite(v) ? v : fallback;
}

const width = parseArgInt('width', 1280);
const height = parseArgInt('height', 720);
const warmup = parseArgInt('warmup', 40);
const iterations = parseArgInt('iterations', 400);

console.log(JSON.stringify(benchmark({ width, height, warmup, iterations })));
