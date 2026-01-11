/* Minimal service worker for PWA-style install + basic caching. */
const CACHE = 'mm-web-app-v1';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) =>
        cache.addAll([
          './',
          './index.html',
          './image_model.onnx',
          './speech_model.onnx',
          './mnist_eval_samples.json',
          './ort-wasm-simd-threaded.mjs',
          './ort-wasm-simd-threaded.jsep.mjs',
          './ort-wasm-simd-threaded.asyncify.mjs',
          './ort-wasm-simd-threaded.wasm',
          './ort-wasm-simd-threaded.jsep.wasm',
          './ort-wasm-simd-threaded.asyncify.wasm',
          './assets/ort-wasm-simd-threaded.mjs',
          './assets/ort-wasm-simd-threaded.jsep.mjs',
          './assets/ort-wasm-simd-threaded.asyncify.mjs',
          './assets/ort-wasm-simd-threaded.wasm',
          './assets/ort-wasm-simd-threaded.jsep.wasm',
          './assets/ort-wasm-simd-threaded.asyncify.wasm',
        ]),
      )
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.map((k) => (k === CACHE ? null : caches.delete(k))))),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
          return res;
        })
        .catch(() => cached);
    }),
  );
});
