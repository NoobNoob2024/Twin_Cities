import * as ort from 'onnxruntime-web';

type SessionState =
  | { status: 'loading' }
  | { status: 'ready'; session: ort.InferenceSession }
  | { status: 'error'; error: string };

let ortConfigured = false;
let imageSessionPromise: Promise<SessionState> | null = null;
let speechSessionPromise: Promise<SessionState> | null = null;

function configureOrtOnce() {
  if (ortConfigured) return;
  ortConfigured = true;

  const baseUrl = import.meta.env.BASE_URL;

  // Load wasm from our static assets (no CDN, no backend).
  // A path prefix works for the wasm EP (it resolves the expected filenames under this prefix).
  ort.env.wasm.wasmPaths = baseUrl;

  const hc = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : 1;
  ort.env.wasm.numThreads = Math.max(1, Math.min(4, hc ?? 1));
}

export async function createImageSession(): Promise<SessionState> {
  if (imageSessionPromise) return imageSessionPromise;
  imageSessionPromise = (async () => {
    configureOrtOnce();
    try {
      const session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}image_model.onnx`, {
        executionProviders: ['wasm'],
      });
      return { status: 'ready', session };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { status: 'error', error: msg };
    }
  })();
  return imageSessionPromise;
}

export async function createSpeechSession(): Promise<SessionState> {
  if (speechSessionPromise) return speechSessionPromise;
  speechSessionPromise = (async () => {
    configureOrtOnce();
    try {
      const session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}speech_model.onnx`, {
        executionProviders: ['wasm'],
      });
      return { status: 'ready', session };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { status: 'error', error: msg };
    }
  })();
  return speechSessionPromise;
}

export type { SessionState };
