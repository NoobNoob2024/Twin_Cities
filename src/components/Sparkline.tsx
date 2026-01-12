import { useEffect, useRef } from 'react';

type Props = {
  values: Float32Array | null;
  title: string;
  height?: number;
};

export function Sparkline({ values, title, height = 64 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = 320;
    const H = height;
    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    if (!values || values.length === 0) return;

    let min = Infinity;
    let max = -Infinity;
    for (const v of values) {
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return;

    const pad = 6;
    const innerH = H - pad * 2;
    const mid = pad + innerH / 2;
    const scale = innerH / (max - min || 1);

    ctx.strokeStyle = 'rgba(37, 99, 235, 0.85)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = pad + (i / (values.length - 1)) * (W - pad * 2);
      const y = mid - (values[i]! - (min + max) / 2) * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.10)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, mid);
    ctx.lineTo(W - pad, mid);
    ctx.stroke();
  }, [height, values]);

  return (
    <div>
      <div className="resultLabel">{title}</div>
      <canvas ref={canvasRef} className="sparklineCanvas" />
    </div>
  );
}
