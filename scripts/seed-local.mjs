// ローカル開発用: 偽の写真と索引を作って、wrangler dev のローカル R2 に入れる。
//   npm run seed:local
// 人物 A（30 枚）・人物 B（20 枚）・その他、の写真 80 枚を作る。
// 人物 A の検索用特徴量を .seed/query-a.json に書き出す（curl で /api/search を試す用）。

import { execFileSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const OUT = '.seed';
const DIM = 512;
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

// ---------------- 乱数（再現できるように固定シード）
let seed = 42;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const unit = (v) => {
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};
const randVec = () => unit(Array.from({ length: DIM }, gauss));
const near = (base, noise) => unit(base.map((x) => x + gauss() * noise));

// ---------------- 最小の PNG エンコーダ
const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function png(w, h, pixel) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const [r, g, b] = pixel(x / w, y / h);
      const o = y * (w * 3 + 1) + 1 + x * 3;
      raw[o] = r;
      raw[o + 1] = g;
      raw[o + 2] = b;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
const palette = [
  [240, 117, 87],
  [53, 178, 127],
  [114, 163, 214],
  [233, 185, 73],
];
function photo(n, w, h, who) {
  const a = palette[n % 4];
  const b = palette[(n + 1) % 4];
  const dot = who === 'A' ? [32, 34, 31] : who === 'B' ? [248, 245, 233] : [150, 150, 150];
  return png(w, h, (x, y) => {
    const t = (x + y) / 2;
    const dx = (x - 0.5) * (w / h);
    const dy = y - 0.45;
    if (dx * dx + dy * dy < 0.03) return dot;
    return a.map((c, i) => Math.round(c * (1 - t) + b[i] * t));
  });
}

// ---------------- 写真・顔・索引
const A = randVec();
const B = randVec();
const photos = [];
const faces = [];
const pairs = [];
for (let n = 0; n < 80; n++) {
  const who = n < 30 ? 'A' : n < 50 ? 'B' : 'X';
  const portrait = n % 5 === 3;
  const [w, h] = portrait ? [2592, 3888] : [3888, 2592];
  const id = createHash('sha256').update(`seed-${n}`).digest('hex').slice(0, 16);
  const name = `DSCF${8400 + n}.JPG`;
  const scale = (s) => (portrait ? [Math.round((s * 2) / 3), s] : [s, Math.round((s * 2) / 3)]);
  const files = {
    [`thumb/${id}.webp`]: photo(n, ...scale(400), who),
    [`large/${id}.webp`]: photo(n, ...scale(1200), who),
    [`orig/${id}.jpg`]: photo(n, ...scale(1600), who),
  };
  for (const [key, buf] of Object.entries(files)) {
    const file = `${OUT}/${key.replace('/', '_')}`;
    writeFileSync(file, buf);
    pairs.push({ key, file });
  }
  photos.push({ id, name, width: w, height: h, bytes: files[`orig/${id}.jpg`].length });
  const pi = photos.length - 1;
  // 本人の顔（似ている度合いをばらつかせる）+ 他の人 0〜2 人
  if (who === 'A') faces.push([pi, near(A, n < 24 ? 0.03 : 0.07)]);
  if (who === 'B') faces.push([pi, near(B, 0.04)]);
  for (let k = 0; k < n % 3; k++) faces.push([pi, randVec()]);
}

const q8 = (rows) => {
  const out = new Int8Array(rows.length * DIM);
  rows.forEach((v, i) => v.forEach((x, k) => (out[i * DIM + k] = Math.max(-127, Math.min(127, Math.round(x * 127))))));
  return out;
};
writeFileSync(`${OUT}/embeddings.bin`, q8(faces.map(([, v]) => v)));
// 高精度モデルの特徴量の代わり（本物と同じく「同じ人ほど近い」別の空間）
const rot = (v) => unit(v.map((x, k) => x + v[(k + 1) % DIM] * 0.5));
writeFileSync(`${OUT}/embeddings_hq.bin`, q8(faces.map(([, v]) => rot(v))));
writeFileSync(
  `${OUT}/meta.json`,
  JSON.stringify({ version: 'seed', dim: DIM, dimHq: DIM, photos, facePhoto: faces.map(([p]) => p) }),
);
pairs.push({ key: 'index/embeddings.bin', file: `${OUT}/embeddings.bin` });
pairs.push({ key: 'index/embeddings_hq.bin', file: `${OUT}/embeddings_hq.bin` });
pairs.push({ key: 'index/meta.json', file: `${OUT}/meta.json` });
writeFileSync(`${OUT}/pairs.json`, JSON.stringify(pairs));
writeFileSync(`${OUT}/query-a.json`, JSON.stringify({ embedding: near(A, 0.03) }));

// ローカル用シークレット（無ければ作る）
if (!existsSync('.dev.vars')) {
  writeFileSync('.dev.vars', `SITE_PASSWORD=hackathon\nSESSION_SECRET=${randomBytes(32).toString('hex')}\n`);
  console.log('.dev.vars を作りました（ローカル用パスワード: hackathon）');
}

console.log(`写真 ${photos.length} 枚・顔 ${faces.length} 個 → ローカル R2 にアップロード`);
execFileSync('npx', ['wrangler', 'r2', 'bulk', 'put', 'hackathon-photos', '--local', '-f', `${OUT}/pairs.json`], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
