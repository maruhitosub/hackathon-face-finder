// 顔写真 → 512 次元の特徴量。ブラウザ内で計算し、写真は送らない。
// insightface buffalo_s と同じ処理（索引側と特徴量が一致することを確認済み）:
//   1. 検出      det_500m (SCRFD) → 顔の枠と 5 点ランドマーク
//   2. 位置合わせ  5 点を基準位置に合わせて 112x112 に切り出す (norm_crop)
//   3. 認識      w600k_mbf (ArcFace) → L2 正規化

import * as ort from 'onnxruntime-web/wasm';

ort.env.wasm.numThreads = 1; // SharedArrayBuffer (COOP/COEP) 無しで動かす

const MODEL_DIR = '/models/';
const DET_SIZE = 640;
const DET_THRESH = 0.5;
const NMS_THRESH = 0.4;
const MAX_SIDE = 1600; // 大きい写真は縮めてから検出する
// ArcFace の基準ランドマーク (112x112)
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

export interface Face {
  box: [number, number, number, number];
  kps: number[][]; // 5 点 [x, y]
  score: number;
}

export interface FaceResult {
  embedding: Float32Array;
  preview: string; // 検索に使った顔の切り抜き (dataURL)
  faceCount: number;
}

export class NoFaceError extends Error {}

/** RGBA 画素 → [1, 3, size, size] の float テンソル ((値 - 127.5) / std) */
function toTensor(rgba: Uint8ClampedArray, size: number, std: number): ort.Tensor {
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) data[ch * n + i] = (rgba[i * 4 + ch] - 127.5) / std;
  }
  return new ort.Tensor('float32', data, [1, 3, size, size]);
}

let sessions: Promise<{ det: ort.InferenceSession; rec: ort.InferenceSession }> | null = null;

export function loadModels() {
  if (!sessions) {
    const opt: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    sessions = Promise.all([
      ort.InferenceSession.create(`${MODEL_DIR}det_500m.onnx`, opt),
      ort.InferenceSession.create(`${MODEL_DIR}w600k_mbf.onnx`, opt),
    ]).then(([det, rec]) => ({ det, rec }));
    sessions.catch(() => (sessions = null));
  }
  return sessions;
}

export async function loadImage(file: Blob): Promise<ImageData> {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s);
  const h = Math.round(bmp.height * s);
  const c = new OffscreenCanvas(w, h);
  const g = c.getContext('2d')!;
  g.imageSmoothingQuality = 'high';
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  return g.getImageData(0, 0, w, h);
}

async function detect(det: ort.InferenceSession, img: ImageData): Promise<Face[]> {
  // 長辺を 640 に合わせ、左上に詰めて 640x640 に（残りは黒）
  const ratio = img.height / img.width;
  const nh = ratio > 1 ? DET_SIZE : Math.floor(DET_SIZE * ratio);
  const nw = ratio > 1 ? Math.floor(DET_SIZE / ratio) : DET_SIZE;
  const scale = nh / img.height;

  const src = new OffscreenCanvas(img.width, img.height);
  src.getContext('2d')!.putImageData(img, 0, 0);
  const c = new OffscreenCanvas(DET_SIZE, DET_SIZE);
  const g = c.getContext('2d')!;
  g.fillStyle = '#000';
  g.fillRect(0, 0, DET_SIZE, DET_SIZE);
  g.drawImage(src, 0, 0, nw, nh);
  const px = g.getImageData(0, 0, DET_SIZE, DET_SIZE).data;

  const out = await det.run({ [det.inputNames[0]]: toTensor(px, DET_SIZE, 128) });
  const o = det.outputNames.map((k) => out[k].data as Float32Array);

  // 出力: scores(8,16,32), bbox(8,16,32), kps(8,16,32)。各位置にアンカー 2 個
  const faces: Face[] = [];
  [8, 16, 32].forEach((stride, si) => {
    const scores = o[si];
    const bbox = o[si + 3];
    const kps = o[si + 6];
    const fw = DET_SIZE / stride;
    for (let a = 0; a < scores.length; a++) {
      if (scores[a] < DET_THRESH) continue;
      const cell = a >> 1;
      const cx = (cell % fw) * stride;
      const cy = Math.floor(cell / fw) * stride;
      const b = bbox.subarray(a * 4, a * 4 + 4);
      const k = kps.subarray(a * 10, a * 10 + 10);
      faces.push({
        score: scores[a],
        box: [(cx - b[0] * stride) / scale, (cy - b[1] * stride) / scale, (cx + b[2] * stride) / scale, (cy + b[3] * stride) / scale],
        kps: [0, 1, 2, 3, 4].map((j) => [(cx + k[j * 2] * stride) / scale, (cy + k[j * 2 + 1] * stride) / scale]),
      });
    }
  });
  return nms(faces);
}

function nms(faces: Face[]): Face[] {
  faces.sort((a, b) => b.score - a.score);
  const keep: Face[] = [];
  const area = (f: Face) => (f.box[2] - f.box[0] + 1) * (f.box[3] - f.box[1] + 1);
  for (const f of faces) {
    const overlaps = keep.some((k) => {
      const w = Math.max(0, Math.min(f.box[2], k.box[2]) - Math.max(f.box[0], k.box[0]) + 1);
      const h = Math.max(0, Math.min(f.box[3], k.box[3]) - Math.max(f.box[1], k.box[1]) + 1);
      return (w * h) / (area(f) + area(k) - w * h) > NMS_THRESH;
    });
    if (!overlaps) keep.push(f);
  }
  return keep;
}

/** 5 点 → 基準位置 の相似変換 (最小二乗, Umeyama と同じ解)。[a, b, tx, ty]: x' = a x - b y + tx, y' = b x + a y + ty */
export function similarity(src: number[][], dst: number[][]): [number, number, number, number] {
  const n = src.length;
  const sx = src.reduce((s, p) => s + p[0], 0) / n;
  const sy = src.reduce((s, p) => s + p[1], 0) / n;
  const dx = dst.reduce((s, p) => s + p[0], 0) / n;
  const dy = dst.reduce((s, p) => s + p[1], 0) / n;
  let num_r = 0;
  let num_i = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const zr = src[i][0] - sx;
    const zi = src[i][1] - sy;
    const wr = dst[i][0] - dx;
    const wi = dst[i][1] - dy;
    num_r += wr * zr + wi * zi; // w * conj(z)
    num_i += wi * zr - wr * zi;
    den += zr * zr + zi * zi;
  }
  const a = num_r / den;
  const b = num_i / den;
  return [a, b, dx - (a * sx - b * sy), dy - (b * sx + a * sy)];
}

/** cv2.warpAffine(img, M, (112,112)) と同じ: 出力画素ごとに逆変換して双線形補間。範囲外は 0 */
export function warp112(img: ImageData, kps: number[][]): ImageData {
  const [a, b, tx, ty] = similarity(kps, ARCFACE_DST);
  const det = a * a + b * b;
  const out = new ImageData(112, 112);
  const { data, width, height } = img;
  for (let y = 0; y < 112; y++) {
    for (let x = 0; x < 112; x++) {
      const u = x - tx;
      const v = y - ty;
      const sx = (a * u + b * v) / det;
      const sy = (-b * u + a * v) / det;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const o = (y * 112 + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        let acc = 0;
        for (const [xx, yy, wgt] of [
          [x0, y0, (1 - fx) * (1 - fy)],
          [x0 + 1, y0, fx * (1 - fy)],
          [x0, y0 + 1, (1 - fx) * fy],
          [x0 + 1, y0 + 1, fx * fy],
        ]) {
          if (xx >= 0 && yy >= 0 && xx < width && yy < height) acc += data[(yy * width + xx) * 4 + ch] * wgt;
        }
        out.data[o + ch] = Math.round(acc);
      }
      out.data[o + 3] = 255;
    }
  }
  return out;
}

async function embed(rec: ort.InferenceSession, face: ImageData): Promise<Float32Array> {
  const out = await rec.run({ [rec.inputNames[0]]: toTensor(face.data, 112, 127.5) });
  const v = Float32Array.from(out[rec.outputNames[0]].data as Float32Array);
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

export async function extractFace(img: ImageData, onStatus?: (s: string) => void): Promise<FaceResult> {
  onStatus?.('顔認識モデルを読み込んでいます（初回のみ・約 16MB）…');
  const { det, rec } = await loadModels();
  onStatus?.('顔を読み取っています…');
  const faces = await detect(det, img);
  if (!faces.length) throw new NoFaceError();
  const face = faces.reduce((a, f) =>
    (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]) > (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) ? f : a,
  );
  const aligned = warp112(img, face.kps);
  const embedding = await embed(rec, aligned);

  const c = new OffscreenCanvas(112, 112);
  c.getContext('2d')!.putImageData(aligned, 0, 0);
  const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
  const preview = await new Promise<string>((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result as string);
    r.readAsDataURL(blob);
  });
  return { embedding, preview, faceCount: faces.length };
}
