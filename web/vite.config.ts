import { defineConfig } from 'vite';

export default defineConfig({
  // 事前バンドルすると onnxruntime-web が wasm を見つけられない
  optimizeDeps: { exclude: ['onnxruntime-web', 'onnxruntime-web/wasm'] },
  server: {
    // API は wrangler dev へ
    proxy: { '/api': 'http://localhost:8787', '/img': 'http://localhost:8787', '/dl': 'http://localhost:8787' },
  },
});
