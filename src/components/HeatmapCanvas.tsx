import { useEffect, useMemo, useRef } from 'react';

type Props = {
  data: Float32Array | null;
  rows: number;
  cols: number;
  title: string;
  mode?: 'mono' | 'diverging';
  animateScan?: boolean;
  className?: string;
};

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function colorFor(value: number, vmin: number, vmax: number, mode: 'mono' | 'diverging') {
  if (!Number.isFinite(value)) return 'rgba(0,0,0,0)';
  const t = (value - vmin) / (vmax - vmin || 1);
  const x = clamp(t, 0, 1);

  if (mode === 'mono') {
    // blue-ish
    const r = Math.round(37 + 20 * x);
    const g = Math.round(99 + 80 * x);
    const b = Math.round(235 + 10 * x);
    return `rgb(${r},${g},${b})`;
  }

  // diverging: negative -> red, positive -> blue
  const mid = 0.5;
  if (x < mid) {
    const a = x / mid;
    const r = Math.round(220);
    const g = Math.round(38 + 120 * a);
    const b = Math.round(38 + 120 * a);
    return `rgb(${r},${g},${b})`;
  }
  const a = (x - mid) / mid;
  const r = Math.round(38 + 120 * (1 - a));
  const g = Math.round(99 + 80 * (1 - a));
  const b = Math.round(235);
  return `rgb(${r},${g},${b})`;
}

export function HeatmapCanvas({ data, rows, cols, title, mode = 'mono', animateScan = false, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    return { min, max };
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = cols;
    const H = rows;
    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = false;

    if (!data || !stats) return;

    const img = ctx.createImageData(W, H);
    const vmin = stats.min;
    const vmax = stats.max;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const v = data[i] ?? 0;
        const colStr = colorFor(v, vmin, vmax, mode);
        // parse rgb(...)
        const m = colStr.match(/rgb\\((\\d+),(\\d+),(\\d+)\\)/);
        const rr = m ? Number(m[1]) : 0;
        const gg = m ? Number(m[2]) : 0;
        const bb = m ? Number(m[3]) : 0;
        const o = i * 4;
        img.data[o] = rr;
        img.data[o + 1] = gg;
        img.data[o + 2] = bb;
        img.data[o + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);
  }, [cols, data, mode, rows, stats]);

  useEffect(() => {
    if (!animateScan) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    const start = performance.now();
    const draw = (t: number) => {
      const elapsed = t - start;
      const period = 1400;
      const p = (elapsed % period) / period;
      const y = Math.floor(p * rows);
      ctx.save();
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(37, 99, 235, 0.18)';
      ctx.fillRect(0, y, cols, 1);
      ctx.restore();
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [animateScan, cols, rows]);

  return (
    <div className={className}>
      <div className="resultLabel">{title}</div>
      <canvas ref={canvasRef} className="heatmapCanvas" />
      {stats && (
        <div className="heatmapMeta mono">
          min={stats.min.toFixed(3)} max={stats.max.toFixed(3)}
        </div>
      )}
    </div>
  );
}
