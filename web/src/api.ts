// バックエンド (Cloudflare Worker) との約束事。VITE_MOCK=1 のときはブラウザ内の偽データで動く。
//
//   POST /api/login   { password }            → 204 (Cookie 発行) / 401
//   POST /api/logout                          → 204
//   GET  /api/session                         → 204 / 401
//   POST /api/search  { embedding: number[] } → SearchResponse / 401
//   GET  /img/thumb/{id}   一覧用 (長辺 400px)
//   GET  /img/large/{id}   拡大表示用 (長辺 1600px)
//   GET  /dl/{id}          元解像度 (Content-Disposition: attachment)

export type Level = 'sure' | 'maybe';

export interface Photo {
  id: string;
  name: string; // 元のファイル名 (DSCF8456.JPG)
  score: number; // 類似度 0〜1
  level: Level;
  width: number;
  height: number;
  bytes: number; // 元画像のサイズ
}

export interface SearchResponse {
  photos: Photo[];
  total: number; // 写真の総数 (2,814)
}

export class AuthError extends Error {}

const MOCK = import.meta.env.VITE_MOCK === '1';

export const urls = {
  thumb: (p: Photo) => (MOCK ? mock.image(p, 400) : `/img/thumb/${p.id}`),
  large: (p: Photo) => (MOCK ? mock.image(p, 1600) : `/img/large/${p.id}`),
  original: (p: Photo) => (MOCK ? mock.image(p, 2400) : `/dl/${p.id}`),
};

async function call(path: string, init?: RequestInit): Promise<Response> {
  const r = await fetch(path, { credentials: 'same-origin', ...init });
  if (r.status === 401) throw new AuthError();
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r;
}

export async function login(password: string): Promise<boolean> {
  if (MOCK) return mock.login(password);
  try {
    await call('/api/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ password }),
    });
    return true;
  } catch (e) {
    if (e instanceof AuthError) return false;
    throw e;
  }
}

export async function logout(): Promise<void> {
  if (MOCK) return mock.logout();
  await call('/api/logout', { method: 'POST' });
}

export async function hasSession(): Promise<boolean> {
  if (MOCK) return mock.loggedIn();
  try {
    await call('/api/session');
    return true;
  } catch (e) {
    if (e instanceof AuthError) return false;
    throw e;
  }
}

export async function search(embedding: Float32Array): Promise<SearchResponse> {
  if (MOCK) return mock.search();
  const r = await call('/api/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ embedding: Array.from(embedding) }),
  });
  return r.json();
}

// ---------------------------------------------------------------- mock
const mock = {
  login(pw: string) {
    const ok = pw === 'hackathon';
    if (ok) sessionStorage.setItem('mock-session', '1');
    return new Promise<boolean>((res) => setTimeout(() => res(ok), 400));
  },
  logout() {
    sessionStorage.removeItem('mock-session');
    return Promise.resolve();
  },
  loggedIn() {
    try {
      return Promise.resolve(sessionStorage.getItem('mock-session') === '1');
    } catch {
      return Promise.resolve(false);
    }
  },
  search(): Promise<SearchResponse> {
    const photos: Photo[] = Array.from({ length: 41 }, (_, i) => {
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
    return new Promise((res) => setTimeout(() => res({ photos, total: 2814 }), 900));
  },
  cache: new Map<string, string>(),
  // 写真の代わりに色面と番号を描いた画像を作る
  image(p: Photo, size: number): string {
    const key = `${p.id}:${size}`;
    const hit = mock.cache.get(key);
    if (hit) return hit;
    const scale = size / Math.max(p.width, p.height);
    const w = Math.round(p.width * scale);
    const h = Math.round(p.height * scale);
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const g = c.getContext('2d')!;
    const palette = ['#f07557', '#35b27f', '#72a3d6', '#e9b949'];
    const n = parseInt(p.id.slice(1), 10);
    const grad = g.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, palette[n % 4]);
    grad.addColorStop(1, palette[(n + 1) % 4]);
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
    const url = c.toDataURL('image/jpeg', 0.85);
    mock.cache.set(key, url);
    return url;
  },
};
