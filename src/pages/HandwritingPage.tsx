import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import * as ort from 'onnxruntime-web';
import { HeatmapCanvas } from '../components/HeatmapCanvas';
import { Sparkline } from '../components/Sparkline';
import { createImageSession } from '../lib/onnx';
import { preprocessMnistFromCanvas } from '../lib/preprocessMnist';

type Prob = { digit: number; p: number };

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / (sum || 1));
}

export function HandwritingPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewHostRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const isDrawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const historyRef = useRef<ImageData[]>([]);

  const [lineWidth, setLineWidth] = useState(18);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState<string>('');

  const [prediction, setPrediction] = useState<number | null>(null);
  const [probs, setProbs] = useState<Prob[]>([]);
  const [busy, setBusy] = useState(false);
  const [inputTensorData, setInputTensorData] = useState<Float32Array | null>(null);
  const [featureVec, setFeatureVec] = useState<Float32Array | null>(null);
  const [vizOn, setVizOn] = useState(true);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraErr, setCameraErr] = useState('');
  const [permDiag, setPermDiag] = useState<Record<string, string>>({});

  const statusDotClass = useMemo(() => {
    if (modelStatus === 'ready') return 'statusDot ok';
    if (modelStatus === 'error') return 'statusDot bad';
    return 'statusDot';
  }, [modelStatus]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setModelStatus('loading');
      const state = await createImageSession();
      if (cancelled) return;
      if (state.status === 'ready') {
        setSession(state.session);
        setModelStatus('ready');
      } else if (state.status === 'error') {
        setModelStatus('error');
        setModelError(state.error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  useEffect(() => {
    const host = previewHostRef.current;
    if (!host) return;
    host.replaceChildren();
  }, []);

  async function refreshPermissions() {
    const diag: Record<string, string> = {};
    diag.protocol = location.protocol;
    diag.isSecureContext = String(window.isSecureContext);
    diag.mediaDevices = String(Boolean(navigator.mediaDevices?.getUserMedia));
    try {
      const p = (navigator.permissions as unknown as { query?: (d: { name: string }) => Promise<{ state: string }> })
        .query;
      if (!p) {
        diag.permissionsApi = 'not supported';
      } else {
        const cam = await p({ name: 'camera' });
        diag.camera = cam.state;
        diag.permissionsApi = 'ok';
      }
    } catch (e) {
      diag.permissionsApi = e instanceof Error ? e.message : String(e);
    }
    setPermDiag(diag);
  }

  useEffect(() => {
    void refreshPermissions();
  }, []);

  async function openCamera() {
    setCameraErr('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      videoStreamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play();
      }
      setCameraOpen(true);
      await refreshPermissions();
    } catch (e) {
      setCameraErr(e instanceof Error ? e.message : String(e));
      await refreshPermissions();
    }
  }

  function closeCamera() {
    setCameraOpen(false);
    const stream = videoStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    videoStreamRef.current = null;
    const v = videoRef.current;
    if (v) v.srcObject = null;
  }

  async function captureFromCamera() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!video || !canvas || !ctx) return;

    snapshotForUndo();

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const size = Math.min(vw, vh);
    const sx = Math.floor((vw - size) / 2);
    const sy = Math.floor((vh - size) / 2);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(video, sx, sy, size, size, 0, 0, canvas.width, canvas.height);

    setPrediction(null);
    setProbs([]);
    setInputTensorData(null);
    setFeatureVec(null);
    const host = previewHostRef.current;
    if (host) host.replaceChildren();
  }

  function getCanvasPos(e: PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function snapshotForUndo() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;
    historyRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (historyRef.current.length > 20) historyRef.current.shift();
  }

  function onPointerDown(e: PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    canvas.setPointerCapture(e.pointerId);
    isDrawingRef.current = true;
    snapshotForUndo();
    const pos = getCanvasPos(e);
    lastPosRef.current = pos;
    if (!pos) return;
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function onPointerMove(e: PointerEvent<HTMLCanvasElement>) {
    if (!isDrawingRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    const pos = getCanvasPos(e);
    const last = lastPosRef.current;
    if (!pos || !last) return;
    ctx.strokeStyle = 'black';
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastPosRef.current = pos;
  }

  function stopDrawing() {
    isDrawingRef.current = false;
    lastPosRef.current = null;
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    historyRef.current = [];
    setPrediction(null);
    setProbs([]);
    setInputTensorData(null);
    setFeatureVec(null);
    const host = previewHostRef.current;
    if (host) host.replaceChildren();
  }

  async function loadImageToCanvas(file: File) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;

    snapshotForUndo();

    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.decoding = 'async';
      img.src = url;
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load image'));
      });

      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Fit image into canvas with padding.
      const pad = Math.round(canvas.width * 0.1);
      const maxW = canvas.width - pad * 2;
      const maxH = canvas.height - pad * 2;
      const scale = Math.min(maxW / img.width, maxH / img.height);
      const w = Math.max(1, Math.round(img.width * scale));
      const h = Math.max(1, Math.round(img.height * scale));
      const x = Math.floor((canvas.width - w) / 2);
      const y = Math.floor((canvas.height - h) / 2);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, x, y, w, h);

      // Reset results; user can press "開始辨識".
      setPrediction(null);
      setProbs([]);
      const host = previewHostRef.current;
      if (host) host.replaceChildren();
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function onPickImage() {
    fileInputRef.current?.click();
  }

  function undo() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !ctx) return;
    const prev = historyRef.current.pop();
    if (!prev) return;
    ctx.putImageData(prev, 0, 0);
  }

  async function recognize() {
    if (!session || !canvasRef.current) return;
    setBusy(true);
    try {
      const { tensorData, previewCanvas, hasInk } = preprocessMnistFromCanvas(canvasRef.current);
      const host = previewHostRef.current;
      if (host) host.replaceChildren(previewCanvas);
      setInputTensorData(tensorData);
      if (!hasInk) {
        setPrediction(null);
        setProbs([]);
        setFeatureVec(null);
        return;
      }

      const inputTensor = new ort.Tensor('float32', tensorData, [1, 1, 28, 28]);
      const results = await session.run({ input: inputTensor });
      const logits = results.output.data as Float32Array;
      const features = results.features?.data as Float32Array | undefined;
      if (features) setFeatureVec(new Float32Array(features));
      const p = softmax(logits);
      const ranked = p
        .map((v, digit) => ({ digit, p: v }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5);
      setProbs(ranked);
      setPrediction(ranked[0]?.digit ?? null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="twoCol">
        <section className="card">
          <div className="cardHeader">
            <div>
              <div className="cardTitle">手寫數字辨識</div>
              <div className="cardHint">在方框內寫 0–9，或上傳圖片（支援滑鼠 / 觸控 / 觸控筆）</div>
            </div>
            <div className="pill" title={modelStatus === 'error' ? modelError : ''}>
              <span className={statusDotClass} aria-hidden />
              <span>模型：{modelStatus === 'loading' ? '載入中' : modelStatus === 'ready' ? '就緒' : '錯誤'}</span>
            </div>
          </div>
          <div className="cardBody">
            <div className="drawWrap">
              <canvas
                ref={canvasRef}
                width={320}
                height={320}
                className="drawCanvas"
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={stopDrawing}
                onPointerCancel={stopDrawing}
                onPointerLeave={stopDrawing}
              />
            </div>

            <div className="controls">
              <div className="btnRow">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void loadImageToCanvas(f);
                    e.currentTarget.value = '';
                  }}
                />
                <button type="button" className="btn" onClick={onPickImage} disabled={busy}>
                  上傳圖片
                </button>
                <button type="button" className="btn" onClick={() => void openCamera()} disabled={busy}>
                  開啟相機
                </button>
                <button type="button" className="btn" onClick={() => setVizOn((v) => !v)} disabled={busy}>
                  {vizOn ? '關閉可視化' : '開啟可視化'}
                </button>
                <button type="button" className="btn" onClick={undo} disabled={busy}>
                  復原
                </button>
                <button type="button" className="btn" onClick={clearCanvas} disabled={busy}>
                  清除
                </button>
                <button
                  type="button"
                  className="btn btnPrimary"
                  onClick={recognize}
                  disabled={!session || modelStatus !== 'ready' || busy}
                >
                  {busy ? '辨識中…' : '開始辨識'}
                </button>
                <span className="pill">
                  筆畫粗細
                  <input
                    className="slider"
                    type="range"
                    min={8}
                    max={28}
                    value={lineWidth}
                    onChange={(e) => setLineWidth(Number(e.target.value))}
                    aria-label="筆畫粗細"
                  />
                  <span className="mono">{lineWidth}</span>
                </span>
              </div>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="cardHeader">
            <div>
              <div className="cardTitle">辨識結果</div>
              <div className="cardHint">模型：ONNX Runtime Web（WASM）｜可視化：輸入張量與特徵向量（模型真實使用）</div>
            </div>
            <div className="resultBig" aria-label="預測結果">
              {prediction === null ? '—' : prediction}
            </div>
          </div>
          <div className="cardBody">
            <div className="resultGrid">
              <div className="resultBlock">
                <div className="resultLabel">28×28 預處理預覽</div>
                <div className="previewHost" ref={previewHostRef} />
              </div>
              <div className="resultBlock">
                <div className="resultLabel">Top-5 機率</div>
                <div className="bars">
                  {probs.length === 0 && <div className="muted">尚未辨識</div>}
                  {probs.map((x) => (
                    <div key={x.digit} className="barRow">
                      <div className="barDigit">{x.digit}</div>
                      <div className="barTrack" aria-hidden>
                        <div className="barFill" style={{ width: `${Math.round(x.p * 100)}%` }} />
                      </div>
                      <div className="barPct mono">{Math.round(x.p * 100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {vizOn && (
              <div className="resultGrid" style={{ marginTop: 14 }}>
                <HeatmapCanvas
                  title="輸入張量熱圖（28×28，Normalize 後）"
                  data={inputTensorData}
                  rows={28}
                  cols={28}
                  mode="diverging"
                  animateScan={busy}
                />
                <Sparkline title="影像特徵向量（encoder 輸出，128 維）" values={featureVec} />
              </div>
            )}
            {modelStatus === 'error' && (
              <div className="errorBox">
                <div className="errorTitle">模型載入失敗</div>
                <div className="errorMsg mono">{modelError || 'Unknown error'}</div>
                <div className="errorHint">
                  建議使用 `npm run dev` 或 `npm run preview` 以 HTTP 方式提供靜態檔案（不要用 file://）。
                </div>
              </div>
            )}

            <div className="card" style={{ marginTop: 14, boxShadow: 'none' }}>
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">相機/環境診斷</div>
                  <div className="cardHint">相機通常需要 HTTPS；若被拒絕需到瀏覽器網站設定允許</div>
                </div>
                <button type="button" className="btn" onClick={refreshPermissions}>
                  重新檢查
                </button>
              </div>
              <div className="cardBody">
                <div className="diagGrid">
                  <div className="kv">
                    <div className="kvKey">protocol</div>
                    <div className="mono">{permDiag.protocol ?? '-'}</div>
                    <div className="kvKey">isSecureContext</div>
                    <div className="mono">{permDiag.isSecureContext ?? '-'}</div>
                    <div className="kvKey">mediaDevices</div>
                    <div className="mono">{permDiag.mediaDevices ?? '-'}</div>
                    <div className="kvKey">camera</div>
                    <div className="mono">{permDiag.camera ?? '-'}</div>
                  </div>
                  <div className="kv">
                    <div className="kvKey">提示</div>
                    <div>
                      <div>1) GitHub Pages：Settings → Pages → 勾選 Enforce HTTPS</div>
                      <div>2) 瀏覽器網址列左側圖示 → Site settings → 允許相機</div>
                      <div>3) 若在 App 內嵌 WebView，可能被策略禁用</div>
                    </div>
                    <div className="kvKey">Permissions API</div>
                    <div className="mono">{permDiag.permissionsApi ?? '-'}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {cameraOpen && (
        <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="相機">
          <div className="modalCard">
            <div className="cardHeader">
              <div>
                <div className="cardTitle">相機擷取</div>
                <div className="cardHint">按「擷取」會把影像貼到畫布，再用同一模型辨識</div>
              </div>
              <button type="button" className="btn" onClick={closeCamera}>
                關閉
              </button>
            </div>
            <div className="cardBody">
              <video ref={videoRef} playsInline muted className="videoPreview" />
              <div className="btnRow" style={{ marginTop: 12 }}>
                <button type="button" className="btn btnPrimary" onClick={() => void captureFromCamera()}>
                  擷取到畫布
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {cameraErr && (
        <div className="errorBox" style={{ marginTop: 14 }}>
          <div className="errorTitle">相機開啟失敗</div>
          <div className="errorMsg mono">{cameraErr}</div>
        </div>
      )}
    </div>
  );
}
