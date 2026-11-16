// 顔の索引（R2 の index/）を読み込み、受け取った特徴量と総当たりで比べる。
//
// index/meta.json          { version, dim, dimHq?, thresholds?, photos: PhotoMeta[], facePhoto: number[] }
// index/embeddings.bin     Int8Array (顔の数 × dim)。buffalo_s (ブラウザと同じ軽いモデル) の特徴量 × 127
// index/embeddings_hq.bin  Int8Array (顔の数 × dimHq)。同じ顔を antelopev2 (高精度モデル) で計算したもの（任意）
//
// 検索の流れ:
//   1. ブラウザから来た特徴量 (buffalo_s) で全顔と比べ、はっきり本人と言える顔（シード）を選ぶ
//   2a. 高精度の特徴量があれば: シードの高精度特徴量の平均で全顔を探し直す（横顔・小さい顔に強い）
//   2b. 無ければ: シードの buffalo_s 特徴量を足して探し直す
// 高精度モデルはブラウザでは重すぎて動かせないが、索引側で事前計算しておけば照合には使える。

export interface PhotoMeta {
  id: string;
  name: string;
  width: number;
  height: number;
  bytes: number;
}

export interface Thresholds {
  seed: number; // buffalo_s: これ以上ならシード
  sure: number; // buffalo_s のみで判定するときの「ほぼ確実」
  maybe: number; // 同「たぶん」
  hqSure: number; // 高精度: 「ほぼ確実」
  hqMaybe: number; // 高精度: 「たぶん」
}

const DEFAULT_THRESHOLDS: Thresholds = { seed: 0.5, sure: 0.5, maybe: 0.4, hqSure: 0.45, hqMaybe: 0.35 };

interface Meta {
  version: string;
  dim: number;
  dimHq?: number;
  thresholds?: Partial<Thresholds>;
  photos: PhotoMeta[];
  facePhoto: number[]; // 顔 i が写っている写真の番号
}

interface Matrix {
  data: Float32Array; // int8 → float (/127) に展開済み
  dim: number;
}

export interface Index {
  meta: Meta;
  emb: Matrix;
  hq: Matrix | null;
  thresholds: Thresholds;
  byId: Map<string, PhotoMeta>;
}

let cached: Promise<Index> | null = null;
let cachedEtag = '';

async function readMatrix(bucket: R2Bucket, key: string, rows: number, dim: number): Promise<Matrix> {
  const obj = await bucket.get(key);
  if (!obj) throw new Error(`${key} がありません`);
  const raw = new Int8Array(await obj.arrayBuffer());
  if (raw.length !== rows * dim || dim % 4) throw new Error(`${key} のサイズが合いません`);
  const data = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) data[i] = raw[i] / 127;
  return { data, dim };
}

/** isolate ごとに 1 回だけ読む。索引を差し替えたら（meta.json の etag が変われば）読み直す */
export async function loadIndex(bucket: R2Bucket): Promise<Index> {
  const head = await bucket.head('index/meta.json');
  if (!head) throw new Error('index/meta.json がありません');
  if (!cached || head.etag !== cachedEtag) {
    cachedEtag = head.etag;
    cached = (async () => {
      const m = await bucket.get('index/meta.json');
      if (!m) throw new Error('index/meta.json がありません');
      const meta = (await m.json()) as Meta;
      const rows = meta.facePhoto.length;
      const [emb, hq] = await Promise.all([
        readMatrix(bucket, 'index/embeddings.bin', rows, meta.dim),
        meta.dimHq ? readMatrix(bucket, 'index/embeddings_hq.bin', rows, meta.dimHq) : null,
      ]);
      return {
        meta,
        emb,
        hq,
        thresholds: { ...DEFAULT_THRESHOLDS, ...meta.thresholds },
        byId: new Map(meta.photos.map((p) => [p.id, p])),
      };
    })();
    cached.catch(() => (cached = null));
  }
  return cached;
}

/** 全顔との内積。1 万顔 x 512 次元で 1 回 約 6ms */
function scores(m: Matrix, q: Float32Array): Float32Array {
  const { data, dim } = m;
  const n = data.length / dim;
  const out = new Float32Array(n);
  for (let i = 0, off = 0; i < n; i++, off += dim) {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let s3 = 0;
    for (let k = 0; k < dim; k += 4) {
      s0 += data[off + k] * q[k];
      s1 += data[off + k + 1] * q[k + 1];
      s2 += data[off + k + 2] * q[k + 2];
      s3 += data[off + k + 3] * q[k + 3];
    }
    out[i] = s0 + s1 + s2 + s3;
  }
  return out;
}

function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

/** 選んだ顔の特徴量を合計して正規化（start を足す場合はそれも含める） */
function centroid(m: Matrix, rows: number[], start?: Float32Array): Float32Array {
  const c = start ? Float32Array.from(start) : new Float32Array(m.dim);
  for (const i of rows) for (let k = 0; k < m.dim; k++) c[k] += m.data[i * m.dim + k];
  return normalize(c);
}

export interface Hit {
  photo: PhotoMeta;
  score: number;
  level: 'sure' | 'maybe';
}

export function searchIndex(index: Index, query: Float32Array): Hit[] {
  const t = index.thresholds;
  const q = normalize(query);
  const s1 = scores(index.emb, q);

  const seeds: number[] = [];
  for (let i = 0; i < s1.length; i++) if (s1[i] >= t.seed) seeds.push(i);

  // 顔ごとのスコアと、その顔を判定するしきい値
  let score: Float32Array;
  let sure: number;
  let maybe: number;
  if (seeds.length && index.hq) {
    score = scores(index.hq, centroid(index.hq, seeds));
    sure = t.hqSure;
    maybe = t.hqMaybe;
  } else {
    const s2 = seeds.length ? scores(index.emb, centroid(index.emb, seeds, q)) : s1;
    score = s1.map((v, i) => Math.max(v, s2[i]));
    sure = t.sure;
    maybe = t.maybe;
  }

  // 写真ごとに最大スコア（1 枚に同じ人は 1 回しか写らない）
  const best = new Map<number, number>();
  for (let i = 0; i < score.length; i++) {
    if (score[i] < maybe) continue;
    const p = index.meta.facePhoto[i];
    if (score[i] > (best.get(p) ?? -Infinity)) best.set(p, score[i]);
  }
  return [...best]
    .map(([p, s]) => ({
      photo: index.meta.photos[p],
      score: Math.round(s * 1000) / 1000,
      level: (s >= sure ? 'sure' : 'maybe') as Hit['level'],
    }))
    .sort((a, b) => b.score - a.score);
}

/** クライアントから来た特徴量を検証して Float32Array に */
export function parseEmbedding(body: unknown, dim: number): Float32Array | null {
  const e = (body as { embedding?: unknown })?.embedding;
  if (!Array.isArray(e) || e.length !== dim) return null;
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const x = e[i];
    if (typeof x !== 'number' || !Number.isFinite(x)) return null;
    v[i] = x;
  }
  return v;
}
