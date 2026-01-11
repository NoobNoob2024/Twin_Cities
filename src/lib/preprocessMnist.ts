const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

export type PreprocessResult = {
  tensorData: Float32Array;
  previewCanvas: HTMLCanvasElement;
  hasInk: boolean;
};

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

export function preprocessMnistFromCanvas(sourceCanvas: HTMLCanvasElement): PreprocessResult {
  const srcCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!srcCtx) throw new Error('Could not read canvas');

  const { width, height } = sourceCanvas;
  const img = srcCtx.getImageData(0, 0, width, height);
  const data = img.data;

  // Find bounding box of "ink" (black strokes on white background).
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  // Threshold in [0..1] after invert (so ink ~ 1, background ~ 0).
  const threshold = 0.12;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const i = (row + x) * 4;
      const r = data[i] / 255;
      const ink = 1 - r;
      if (ink > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  const previewCanvas = document.createElement('canvas');
  previewCanvas.width = 28;
  previewCanvas.height = 28;
  const previewCtx = previewCanvas.getContext('2d', { willReadFrequently: true });
  if (!previewCtx) throw new Error('Could not create preview canvas');

  // If nothing drawn, return zeroed tensor.
  if (maxX < 0 || maxY < 0) {
    previewCtx.fillStyle = 'black';
    previewCtx.fillRect(0, 0, 28, 28);
    return { tensorData: new Float32Array(28 * 28), previewCanvas, hasInk: false };
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const boxSize = Math.max(boxW, boxH);
  const margin = Math.round(boxSize * 0.2);
  const cropX = clamp(minX - margin, 0, width - 1);
  const cropY = clamp(minY - margin, 0, height - 1);
  const cropMaxX = clamp(maxX + margin, 0, width - 1);
  const cropMaxY = clamp(maxY + margin, 0, height - 1);
  const cropW = cropMaxX - cropX + 1;
  const cropH = cropMaxY - cropY + 1;

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
  if (!cropCtx) throw new Error('Could not create crop canvas');
  cropCtx.drawImage(sourceCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

  // Scale crop to fit 20x20 (MNIST style).
  const targetInkSize = 20;
  const scale = targetInkSize / Math.max(cropW, cropH);
  const scaledW = Math.max(1, Math.round(cropW * scale));
  const scaledH = Math.max(1, Math.round(cropH * scale));

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = scaledW;
  scaledCanvas.height = scaledH;
  const scaledCtx = scaledCanvas.getContext('2d', { willReadFrequently: true });
  if (!scaledCtx) throw new Error('Could not create scaled canvas');
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, 0, 0, scaledW, scaledH);

  // Compose into 28x28 with centering.
  previewCtx.fillStyle = 'white';
  previewCtx.fillRect(0, 0, 28, 28);
  const offsetX = Math.floor((28 - scaledW) / 2);
  const offsetY = Math.floor((28 - scaledH) / 2);
  previewCtx.drawImage(scaledCanvas, offsetX, offsetY);

  const finalData = previewCtx.getImageData(0, 0, 28, 28).data;
  const tensorData = new Float32Array(28 * 28);
  for (let i = 0; i < finalData.length; i += 4) {
    const r = finalData[i] / 255;
    const ink = 1 - r;
    tensorData[i / 4] = (ink - MNIST_MEAN) / MNIST_STD;
  }

  return { tensorData, previewCanvas, hasInk: true };
}
