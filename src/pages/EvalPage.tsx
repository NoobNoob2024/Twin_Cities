import { useEffect, useMemo, useState } from 'react';
import * as ort from 'onnxruntime-web';
import { createImageSession } from '../lib/onnx';

type Sample = { label: number; pixelsB64: string };
type Dataset = { samples: Sample[] };

function b64ToU8(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function argmax(logits: Float32Array) {
  let bestIdx = 0;
  let best = logits[0] ?? -Infinity;
  for (let i = 1; i < logits.length; i++) {
    const v = logits[i]!;
    if (v > best) {
      best = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function EvalPage() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelError, setModelError] = useState('');

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [dataStatus, setDataStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [dataError, setDataError] = useState('');

  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [total, setTotal] = useState(0);
  const [correct, setCorrect] = useState(0);
  const [conf, setConf] = useState<number[][]>(() => Array.from({ length: 10 }, () => Array(10).fill(0)));

  const statusDotClass = useMemo(() => {
    if (modelStatus === 'ready' && dataStatus === 'ready') return 'statusDot ok';
    if (modelStatus === 'error' || dataStatus === 'error') return 'statusDot bad';
    return 'statusDot';
  }, [dataStatus, modelStatus]);

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
    let cancelled = false;
    (async () => {
      setDataStatus('loading');
      try {
        const res = await fetch(`${import.meta.env.BASE_URL}mnist_eval_samples.json`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = (await res.json()) as Dataset;
        if (!json.samples?.length) throw new Error('Empty dataset');
        if (cancelled) return;
        setDataset(json);
        setDataStatus('ready');
        setTotal(json.samples.length);
      } catch (e) {
        if (cancelled) return;
        setDataStatus('error');
        setDataError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function runEval() {
    if (!session || !dataset) return;
    setRunning(true);
    setDone(0);
    setCorrect(0);
    setConf(Array.from({ length: 10 }, () => Array(10).fill(0)));

    // batch = 1 (simple + stable)
    let localCorrect = 0;
    const localConf: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
    for (let i = 0; i < dataset.samples.length; i++) {
      const s = dataset.samples[i]!;
      const pixels = b64ToU8(s.pixelsB64);
      // MNIST is white digit on black bg. Normalize: x = (x/255 - mean) / std.
      const x = new Float32Array(28 * 28);
      for (let j = 0; j < 28 * 28; j++) {
        const v = (pixels[j]! / 255 - 0.1307) / 0.3081;
        x[j] = v;
      }

      const inputTensor = new ort.Tensor('float32', x, [1, 1, 28, 28]);
      const out = await session.run({ input: inputTensor });
      const logits = out.output.data as Float32Array;
      const pred = argmax(logits);
      const y = s.label;

      localConf[y]![pred] = (localConf[y]![pred] ?? 0) + 1;
      if (pred === y) localCorrect++;

      // Update UI every 10 samples to keep it responsive.
      if ((i + 1) % 10 === 0 || i + 1 === dataset.samples.length) {
        setConf(localConf.map((row) => row.slice()));
        setCorrect(localCorrect);
        setDone(i + 1);
      }
    }

    setRunning(false);
  }

  const accuracy = total ? (correct / total) * 100 : 0;

  return (
    <div className="page">
      <section className="card">
        <div className="cardHeader">
          <div>
            <div className="cardTitle">模型準確度展示（手寫）</div>
            <div className="cardHint">使用內建 MNIST 測試樣本（純前端）</div>
          </div>
          <div className="pill" title={[modelError, dataError].filter(Boolean).join(' | ')}>
            <span className={statusDotClass} aria-hidden />
            <span>
              狀態：{modelStatus === 'ready' && dataStatus === 'ready' ? '就緒' : '載入中/錯誤'}
            </span>
          </div>
        </div>
        <div className="cardBody">
          <div className="btnRow">
            <button
              type="button"
              className="btn btnPrimary"
              onClick={runEval}
              disabled={!session || !dataset || running || modelStatus !== 'ready' || dataStatus !== 'ready'}
            >
              {running ? '評估中…' : '開始評估'}
            </button>
            <span className="pill">
              進度：<span className="mono">{done}</span>/<span className="mono">{total}</span>
            </span>
            <span className="pill">
              Accuracy：<span className="mono">{accuracy.toFixed(2)}%</span>
            </span>
          </div>

          {(modelStatus === 'error' || dataStatus === 'error') && (
            <div className="errorBox">
              <div className="errorTitle">載入失敗</div>
              <div className="errorMsg mono">{[modelError, dataError].filter(Boolean).join('\n')}</div>
            </div>
          )}

          <div style={{ marginTop: 14 }}>
            <div className="resultLabel">混淆矩陣（真值 × 預測）</div>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table className="confTable">
                <thead>
                  <tr>
                    <th className="mono">y\\ŷ</th>
                    {Array.from({ length: 10 }, (_, i) => (
                      <th key={i} className="mono">
                        {i}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {conf.map((row, y) => (
                    <tr key={y}>
                      <th className="mono">{y}</th>
                      {row.map((v, x) => (
                        <td key={x} className="mono">
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
