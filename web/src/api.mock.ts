// バックエンド無しで画面を触るための偽 API（npm run dev:mock。パスワード: hackathon）

import type { Api, Photo } from './api';

const SESSION_KEY = 'mock-session';
const PALETTE = ['#f07557', '#35b27f', '#72a3d6', '#e9b949'];
const delay = <T>(value: T, ms: number) => new Promise<T>((res) => setTimeout(() => res(value), ms));

function fakePhotos(): Photo[] {
  return Array.from({ length: 41 }, (_, i) => {
    const score = i < 34 ? 0.92 - i * 0.004 : 0.54 - (i - 34) * 0.012;
    const portrait = i % 5 === 3;
    return {
      id: `p${String(i).padStart(4, '0')}`,
      name: `DSCF${8400 + i * 7}.JPG`,
      score,
      level: score >= 0.55 ? 'sure' : 'maybe',
      width: portrait ? 2592 : 3888,
      height: portrait ? 3888 : 2592,
      bytes: 5_200_000 + ((i * 377_000) % 1_400_000),
    };
  });
}

const imageCache = new Map<string, string>();

/** 写真の代わりに、色面とファイル名を描いた画像 */
function fakeImage(p: Photo, size: number): string {
  const key = `${p.id}:${size}`;
  const hit = imageCache.get(key);
  if (hit) return hit;

  const scale = size / Math.max(p.width, p.height);
  const w = Math.round(p.width * scale);
  const h = Math.round(p.height * scale);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d')!;
  const n = parseInt(p.id.slice(1), 10);
  const grad = g.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, PALETTE[n % 4]);
  grad.addColorStop(1, PALETTE[(n + 1) % 4]);
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(248,245,233,.9)';
  g.beginPath();
  g.arc(w * 0.5, h * 0.42, Math.min(w, h) * 0.16, 0, Math.PI * 2);
  g.fill();
  g.fillRect(w * 0.3, h * 0.62, w * 0.4, h * 0.5);
  g.fillStyle = '#20221f';
  g.font = `900 ${Math.round(Math.min(w, h) * 0.08)}px system-ui, sans-serif`;
  g.fillText(p.name, w * 0.05, h * 0.12);

  const url = canvas.toDataURL('image/jpeg', 0.85);
  imageCache.set(key, url);
  return url;
}

export const mockApi: Api = {
  login(password) {
    const ok = password === 'hackathon';
    if (ok) sessionStorage.setItem(SESSION_KEY, '1');
    return delay(ok, 400);
  },
  async logout() {
    sessionStorage.removeItem(SESSION_KEY);
  },
  async hasSession() {
    try {
      return sessionStorage.getItem(SESSION_KEY) === '1';
    } catch {
      return false;
    }
  },
  search: () => delay({ photos: fakePhotos(), total: 2814 }, 900),
  thumbUrl: (p) => fakeImage(p, 400),
  largeUrl: (p) => fakeImage(p, 1600),
  originalUrl: (p) => fakeImage(p, 2400),
};
