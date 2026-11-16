// 顔の索引 (R2 の index/) と総当たりで照合する。
//
//   meta.json          { version, dim, dimHq?, thresholds?, photos, facePhoto }
//   embeddings.bin     int8 [顔 × dim]    buffalo_s（ブラウザと同じ軽いモデル）
//   embeddings_hq.bin  int8 [顔 × dimHq]  antelopev2（高精度。ブラウザでは重くて動かせない）
//
// 1. ブラウザの特徴量 (buffalo_s) で、確実に本人と言える顔（シード）を選ぶ
// 2. シードの高精度特徴量の平均で全顔を探し直す。高精度が無ければ buffalo_s のまま探し直す

export interface PhotoMeta {
  id: string;
  name: string;
  width: number;
  height: number;
  bytes: number;
}

export interface Thresholds {
  seed: number; // buffalo_s: シードにする下限
  sure: number; // buffalo_s のみで判定するとき
  maybe: number;
  hqSure: number; // 高精度で判定するとき
  hqMaybe: number;
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

/** 行 = 顔。int8 を float (/127) に展開済み */
interface Matrix {
  data: Float32Array;
  dim: number;
}

export interface Index {
  meta: Meta;
  emb: Matrix;
  hq: Matrix | null;
  thresholds: Thresholds;
  byId: Map<string, PhotoMeta>;
}

export interface Hit {
  photo: PhotoMeta;
  score: number;
  level: 'sure' | 'maybe';
}

let cached: Promise<Index> | null = null;
let cachedEtag = '';

/** isolate ごとに 1 回読む。meta.json の etag が変われば読み直す */
export async function loadIndex(bucket: R2Bucket): Promise<Index> {
  const head = await bucket.head('index/meta.json');
  if (!head) throw new Error('index/meta.json がありません');
  if (!cached || head.etag !== cachedEtag) {
    cachedEtag = head.etag;
    cached = readIndex(bucket);
    cached.catch(() => (cached = null));
  }
  return cached;
}

async function readIndex(bucket: R2Bucket): Promise<Index> {
  const obj = await bucket.get('index/meta.json');
  if (!obj) throw new Error('index/meta.json がありません');
  const meta = (await obj.json()) as Meta;
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
}

async function readMatrix(bucket: R2Bucket, key: string, rows: number, dim: number): Promise<Matrix> {
  const obj = await bucket.get(key);
  if (!obj) throw new Error(`${key} がありません`);
  const raw = new Int8Array(await obj.arrayBuffer());
  if (raw.length !== rows * dim || dim % 4) throw new Error(`${key} のサイズが合いません`);
  const data = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) data[i] = raw[i] / 127;
  return { data, dim };
}

/** 全行との内積。1 万顔 × 512 次元で約 6ms（4 本に分けて足すと速い） */
function dotAll({ data, dim }: Matrix, q: Float32Array): Float32Array {
  const out = new Float32Array(data.length / dim);
  for (let i = 0, off = 0; i < out.length; i++, off += dim) {
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
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  return v.map((x) => x / norm);
}

/** 指定した行の合計（start があればそれも足す）を正規化 */
function centroid({ data, dim }: Matrix, rows: number[], start?: Float32Array): Float32Array {
  const c = start ? Float32Array.from(start) : new Float32Array(dim);
  for (const i of rows) for (let k = 0; k < dim; k++) c[k] += data[i * dim + k];
  return normalize(c);
}

export function searchIndex(index: Index, query: Float32Array): Hit[] {
  const t = index.thresholds;
  const q = normalize(query);
  const first = dotAll(index.emb, q);

  const seeds: number[] = [];
  first.forEach((s, i) => s >= t.seed && seeds.push(i));

  let scores = first;
  let [sure, maybe] = [t.sure, t.maybe];
  if (seeds.length && index.hq) {
    scores = dotAll(index.hq, centroid(index.hq, seeds));
    [sure, maybe] = [t.hqSure, t.hqMaybe];
  } else if (seeds.length) {
    const second = dotAll(index.emb, centroid(index.emb, seeds, q));
    scores = first.map((s, i) => Math.max(s, second[i]));
  }

  // 写真ごとに最大スコア
  const best = new Map<number, number>();
  scores.forEach((s, i) => {
    const photo = index.meta.facePhoto[i];
    if (s >= maybe && s > (best.get(photo) ?? -Infinity)) best.set(photo, s);
  });
  return [...best]
    .sort((a, b) => b[1] - a[1])
    .map(([photo, s]) => ({
      photo: index.meta.photos[photo],
      score: Math.round(s * 1000) / 1000,
      level: s >= sure ? 'sure' : 'maybe',
    }));
}

/** 数値 dim 個の配列だけを受け付ける */
export function parseEmbedding(body: unknown, dim: number): Float32Array | null {
  const e = (body as { embedding?: unknown } | null)?.embedding;
  if (!Array.isArray(e) || e.length !== dim) return null;
  if (!e.every((x) => typeof x === 'number' && Number.isFinite(x))) return null;
  return Float32Array.from(e as number[]);
}
