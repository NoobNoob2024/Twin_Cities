import { useEffect, useMemo, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import {
  decodeAudioFileToMono,
  recordMono,
  resampleLinear,
  takeCenterSegment,
  takeOrPadToLength,
} from '../lib/audio';
import { createSpeechSession } from '../lib/onnx';
import { melSpectrogram128x32From16k, SPEECH_INPUT } from '../lib/melSpectrogram';

function extractDigit(text: string): number | null {
  const normalized = text.trim().toLowerCase();

  const map: Record<string, number> = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    零: 0,
    一: 1,
    二: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
    zero: 0,
    one: 1,
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
  };

  if (normalized in map) return map[normalized] ?? null;

  const m = normalized.match(/[0-9]/);
  if (m) return Number(m[0]);
  for (const k of Object.keys(map)) {
    if (k.length === 1 && normalized.includes(k)) return map[k] ?? null;
  }
  return null;
}

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / (sum || 1));
}

export function VoicePage() {
  const [mode, setMode] = useState<'ai' | 'browser'>(() => {
    const saved = localStorage.getItem('mm.voiceMode');
    if (saved === 'ai' || saved === 'browser') return saved;
    return 'ai';
  });

  useEffect(() => {
    localStorage.setItem('mm.voiceMode', mode);
  }, [mode]);

  // --- AI model (ONNX) ---
  const [speechSession, setSpeechSession] = useState<ort.InferenceSession | null>(null);
  const [speechModelStatus, setSpeechModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [speechModelError, setSpeechModelError] = useState('');
  const [aiBusy, setAiBusy] = useState(false);
  const [digit, setDigit] = useState<number | null>(null);
  const [aiProbs, setAiProbs] = useState<Array<{ digit: number; p: number }>>([]);
  const [err, setErr] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setSpeechModelStatus('loading');
      const state = await createSpeechSession();
      if (cancelled) return;
      if (state.status === 'ready') {
        setSpeechSession(state.session);
        setSpeechModelStatus('ready');
      } else if (state.status === 'error') {
        setSpeechModelStatus('error');
        setSpeechModelError(state.error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function runAiOnce() {
    if (!speechSession) return;
    setErr('');
    setDigit(null);
    setAiProbs([]);
    setAiBusy(true);
    try {
      const captured = await recordMono(1050);
      const oneSec = takeOrPadToLength(captured.samples, Math.round(captured.sampleRate * 1.0));
      const resampled = resampleLinear(oneSec, captured.sampleRate, SPEECH_INPUT.sampleRate);
      const fixed16k = takeOrPadToLength(resampled, SPEECH_INPUT.sampleRate);

      const mel = melSpectrogram128x32From16k(fixed16k);
      const inputTensor = new ort.Tensor('float32', mel, [1, 1, SPEECH_INPUT.nMels, SPEECH_INPUT.frames]);
      const results = await speechSession.run({ input: inputTensor });
      const logits = results.output.data as Float32Array;
      const p = softmax(logits);
      const ranked = p
        .map((v, d) => ({ digit: d, p: v }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5);
      setAiProbs(ranked);
      setDigit(ranked[0]?.digit ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setAiBusy(false);
    }
  }

  async function runAiFromFile(file: File) {
    if (!speechSession) return;
    setErr('');
    setDigit(null);
    setAiProbs([]);
    setAiBusy(true);
    try {
      const decoded = await decodeAudioFileToMono(file);
      const resampled = resampleLinear(decoded.samples, decoded.sampleRate, SPEECH_INPUT.sampleRate);
      const centered = takeCenterSegment(resampled, SPEECH_INPUT.sampleRate);
      const fixed16k = takeOrPadToLength(centered, SPEECH_INPUT.sampleRate);

      const mel = melSpectrogram128x32From16k(fixed16k);
      const inputTensor = new ort.Tensor('float32', mel, [1, 1, SPEECH_INPUT.nMels, SPEECH_INPUT.frames]);
      const results = await speechSession.run({ input: inputTensor });
      const logits = results.output.data as Float32Array;
      const p = softmax(logits);
      const ranked = p
        .map((v, d) => ({ digit: d, p: v }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5);
      setAiProbs(ranked);
      setDigit(ranked[0]?.digit ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setAiBusy(false);
    }
  }

  // --- Browser fallback (Web Speech API) ---
  const SpeechRecognitionCtor = useMemo(() => {
    const w = window as unknown as {
      SpeechRecognition?: typeof SpeechRecognition;
      webkitSpeechRecognition?: typeof SpeechRecognition;
    };
    return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
  }, []);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [speechSupported] = useState(Boolean(SpeechRecognitionCtor));
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');

  useEffect(() => {
    if (!SpeechRecognitionCtor) return;
    const rec = new SpeechRecognitionCtor();
    rec.lang = 'zh-TW';
    rec.continuous = false;
    rec.interimResults = true;

    rec.onstart = () => {
      setErr('');
      setListening(true);
      setTranscript('');
      setDigit(null);
    };

    rec.onresult = (ev) => {
      const text = Array.from(ev.results)
        .map((r) => r[0]?.transcript ?? '')
        .join(' ');
      setTranscript(text);
      setDigit(extractDigit(text));
    };

    rec.onerror = (ev) => {
      setErr(ev.error || 'Speech recognition error');
      setListening(false);
    };

    rec.onend = () => {
      setListening(false);
    };

    recognitionRef.current = rec;
    return () => {
      rec.abort();
      recognitionRef.current = null;
    };
  }, [SpeechRecognitionCtor]);

  function startBrowser() {
    try {
      recognitionRef.current?.start();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setListening(false);
    }
  }

  function stopBrowser() {
    recognitionRef.current?.stop();
  }

  return (
    <div className="page">
      <section className="card">
        <div className="cardHeader">
          <div>
            <div className="cardTitle">語音數字辨識</div>
            <div className="cardHint">
              AI 模型：ONNX（MelSpectrogram 128×32）｜備援：Web Speech API（語音轉文字）
            </div>
          </div>
          <div className="pill" title={speechModelStatus === 'error' ? speechModelError : ''}>
            <span
              className={
                speechModelStatus === 'ready'
                  ? 'statusDot ok'
                  : speechModelStatus === 'error'
                    ? 'statusDot bad'
                    : 'statusDot'
              }
              aria-hidden
            />
            <span>
              AI：{speechModelStatus === 'loading' ? '載入中' : speechModelStatus === 'ready' ? '就緒' : '錯誤'}
            </span>
          </div>
        </div>

        <div className="cardBody">
          <div className="segmented" aria-label="語音模式">
            <button
              type="button"
              className={mode === 'ai' ? 'segmentedBtn active' : 'segmentedBtn'}
              onClick={() => setMode('ai')}
            >
              AI 模型辨識
            </button>
            <button
              type="button"
              className={mode === 'browser' ? 'segmentedBtn active' : 'segmentedBtn'}
              onClick={() => setMode('browser')}
            >
              瀏覽器語音轉文字
            </button>
          </div>

          {mode === 'ai' && (
            <div className="voicePanel" style={{ marginTop: 12 }}>
              <div className="btnRow">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void runAiFromFile(f);
                    e.currentTarget.value = '';
                  }}
                />
                <button
                  type="button"
                  className="btn btnPrimary"
                  onClick={runAiOnce}
                  disabled={!speechSession || speechModelStatus !== 'ready' || aiBusy}
                >
                  {aiBusy ? '錄音/辨識中…' : '錄音 1 秒並辨識'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!speechSession || speechModelStatus !== 'ready' || aiBusy}
                >
                  上傳音檔辨識
                </button>
                <span className="pill mono">
                  SR={SPEECH_INPUT.sampleRate}Hz · Mel={SPEECH_INPUT.nMels}×{SPEECH_INPUT.frames}
                </span>
              </div>

              <div className="voiceResult">
                <div className="voiceBlock">
                  <div className="resultLabel">Top-5 機率</div>
                  <div className="bars">
                    {aiProbs.length === 0 && <div className="muted">尚未辨識</div>}
                    {aiProbs.map((x) => (
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
                <div className="voiceBlock">
                  <div className="resultLabel">數字結果</div>
                  <div className="resultBig">{digit === null ? '—' : digit}</div>
                </div>
              </div>

              {speechModelStatus === 'error' && (
                <div className="errorBox">
                  <div className="errorTitle">AI 模型載入失敗</div>
                  <div className="errorMsg mono">{speechModelError || 'Unknown error'}</div>
                </div>
              )}
            </div>
          )}

          {mode === 'browser' && (
            <div className="voicePanel" style={{ marginTop: 12 }}>
              {!speechSupported && (
                <div className="errorBox">
                  <div className="errorTitle">此瀏覽器不支援語音辨識</div>
                  <div className="errorHint">建議使用最新版 Chrome / Edge（桌機或 Android）。</div>
                </div>
              )}

              <div className="btnRow">
                <button type="button" className="btn btnPrimary" onClick={startBrowser} disabled={!speechSupported || listening}>
                  開始聆聽
                </button>
                <button type="button" className="btn" onClick={stopBrowser} disabled={!speechSupported || !listening}>
                  停止
                </button>
                <span className="pill">{listening ? '聆聽中…' : '待命'}</span>
              </div>

              <div className="voiceResult">
                <div className="voiceBlock">
                  <div className="resultLabel">辨識文字</div>
                  <div className={transcript ? 'voiceText' : 'voiceText muted'}>{transcript || '—'}</div>
                </div>
                <div className="voiceBlock">
                  <div className="resultLabel">數字結果</div>
                  <div className="resultBig">{digit === null ? '—' : digit}</div>
                </div>
              </div>
            </div>
          )}

          {err && (
            <div className="errorBox" style={{ marginTop: 12 }}>
              <div className="errorTitle">錯誤</div>
              <div className="errorMsg mono">{err}</div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
