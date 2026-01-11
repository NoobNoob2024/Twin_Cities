import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

// https://vitejs.dev/config/
export default defineConfig({
  // Pure frontend (static) friendly: relative asset paths for sub-path hosting.
  base: './',
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: '.'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.mjs',
          dest: '.'
        },
        // GitHub Pages (and some bundler flows) may request the ORT runtime files under `/assets/`.
        // Copy a non-hashed copy there to avoid 404s like:
        //   /assets/ort-wasm-simd-threaded.jsep.mjs
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'assets'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.mjs',
          dest: 'assets'
        }
      ]
    })
  ],
})
