import { defineConfig } from 'vite';

export default defineConfig({
  // onnxruntime-web は import.meta.url から wasm を探す。事前バンドルすると dev でその場所がずれる
  optimizeDeps: { exclude: ['onnxruntime-web', 'onnxruntime-web/wasm'] },
  server: {
    // npm run dev (モックなし) のとき、API は wrangler dev に流す
    proxy: { '/api': 'http://localhost:8787', '/img': 'http://localhost:8787', '/dl': 'http://localhost:8787' },
  },
});
