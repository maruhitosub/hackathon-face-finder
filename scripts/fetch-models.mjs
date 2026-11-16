// ブラウザで使う顔認識モデルを web/public/models に置く（onnxruntime-web の wasm は Vite が同梱する）。
//   npm run models
// モデル: insightface buffalo_s の det_500m.onnx と w600k_mbf.onnx
// ※ insightface の学習済みモデルは非商用・研究目的のライセンス。

import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const MODELS = 'web/public/models';
const ZIP_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip';
const WANT = ['det_500m.onnx', 'w600k_mbf.onnx'];

mkdirSync(MODELS, { recursive: true });

if (WANT.every((f) => existsSync(join(MODELS, f)))) {
  console.log('モデルは取得済み');
} else {
  const tmp = mkdtempSync(join(tmpdir(), 'buffalo_s-'));
  const zip = join(tmp, 'buffalo_s.zip');
  console.log('buffalo_s.zip をダウンロード中…');
  const res = await fetch(ZIP_URL);
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  writeFileSync(zip, Buffer.from(await res.arrayBuffer()));
  execFileSync('tar', ['-xf', zip, '-C', tmp]); // Windows 10+ の tar も zip を展開できる
  for (const f of WANT) {
    const found = [join(tmp, f), join(tmp, 'buffalo_s', f)].find(existsSync);
    if (!found) throw new Error(`${f} が zip にありません`);
    copyFileSync(found, join(MODELS, f));
  }
  rmSync(tmp, { recursive: true, force: true });
  console.log('モデルを配置:', WANT.join(', '));
}
