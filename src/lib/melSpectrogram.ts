const SAMPLE_RATE = 16000;
const N_FFT = 1024;
const HOP_LENGTH = 512;
const N_MELS = 128;
const N_FRAMES = 32;

function hzToMel(hz: number) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function hannWindow(n: number) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

function bitReversePermutation(re: Float32Array, im: Float32Array) {
  const n = re.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
}

function fftInPlace(re: Float32Array, im: Float32Array) {
  const n = re.length;
  bitReversePermutation(re, im);
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }
}

let cachedWindow: Float32Array | null = null;
let cachedFilters: Float32Array[] | null = null;

function getWindow() {
  cachedWindow ??= hannWindow(N_FFT);
  return cachedWindow;
}

function getMelFilters(): Float32Array[] {
  if (cachedFilters) return cachedFilters;
  const nFftBins = N_FFT / 2 + 1;
  const fMin = 0;
  const fMax = SAMPLE_RATE / 2;
  const mMin = hzToMel(fMin);
  const mMax = hzToMel(fMax);

  const melPoints = new Float32Array(N_MELS + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = mMin + (i * (mMax - mMin)) / (N_MELS + 1);
  }
  const hzPoints = Array.from(melPoints, melToHz);
  const bin = hzPoints.map((hz) => Math.floor((N_FFT + 1) * hz / SAMPLE_RATE));

  const filters: Float32Array[] = [];
  for (let m = 1; m <= N_MELS; m++) {
    const f = new Float32Array(nFftBins);
    const left = bin[m - 1] ?? 0;
    const center = bin[m] ?? 0;
    const right = bin[m + 1] ?? 0;
    if (right <= left) {
      filters.push(f);
      continue;
    }
    for (let k = left; k < center; k++) {
      if (k >= 0 && k < nFftBins) f[k] = (k - left) / Math.max(1, center - left);
    }
    for (let k = center; k < right; k++) {
      if (k >= 0 && k < nFftBins) f[k] = (right - k) / Math.max(1, right - center);
    }
    filters.push(f);
  }
  cachedFilters = filters;
  return filters;
}

export function melSpectrogram128x32From16k(samples16k: Float32Array): Float32Array {
  // center padding (zero) by n_fft/2 on both sides => exactly 32 frames for 1s @16k with hop 512
  const pad = N_FFT / 2;
  const padded = new Float32Array(samples16k.length + pad * 2);
  padded.set(samples16k, pad);

  const window = getWindow();
  const filters = getMelFilters();

  const nFftBins = N_FFT / 2 + 1;
  const out = new Float32Array(N_MELS * N_FRAMES);

  const frame = new Float32Array(N_FFT);
  const re = new Float32Array(N_FFT);
  const im = new Float32Array(N_FFT);
  const power = new Float32Array(nFftBins);

  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      frame[i] = (padded[start + i] ?? 0) * window[i]!;
    }

    re.set(frame);
    im.fill(0);
    fftInPlace(re, im);

    for (let k = 0; k < nFftBins; k++) {
      const r = re[k]!;
      const ii = im[k]!;
      power[k] = r * r + ii * ii;
    }

    for (let m = 0; m < N_MELS; m++) {
      const filt = filters[m]!;
      let sum = 0;
      for (let k = 0; k < nFftBins; k++) sum += power[k]! * filt[k]!;
      out[m * N_FRAMES + t] = sum;
    }
  }

  return out;
}

export const SPEECH_INPUT = {
  sampleRate: SAMPLE_RATE,
  nFft: N_FFT,
  hopLength: HOP_LENGTH,
  nMels: N_MELS,
  frames: N_FRAMES,
} as const;
