// Worker の API。VITE_MOCK=1 のときは api.mock.ts の偽データで動く。
//
//   POST /api/login   { password }            → 204 / 401
//   POST /api/logout                          → 204
//   GET  /api/session                         → 204 / 401
//   POST /api/search  { embedding: number[] } → SearchResponse
//   GET  /img/thumb/{id}, /img/large/{id}     一覧用 400px / 拡大用 1600px
//   GET  /dl/{id}                             元画像

import { mockApi } from './api.mock';

export type Level = 'sure' | 'maybe';

export interface Photo {
  id: string;
  name: string; // 元のファイル名
  score: number; // 類似度 0〜1
  level: Level;
  width: number;
  height: number;
  bytes: number; // 元画像のサイズ
}

export interface SearchResponse {
  photos: Photo[];
  total: number; // 写真の総数
}

export interface Api {
  login(password: string): Promise<boolean>;
  logout(): Promise<void>;
  hasSession(): Promise<boolean>;
  search(embedding: Float32Array): Promise<SearchResponse>;
  thumbUrl(p: Photo): string;
  largeUrl(p: Photo): string;
  originalUrl(p: Photo): string;
}

/** セッション切れ (401) */
export class AuthError extends Error {}

async function call(path: string, body?: unknown): Promise<Response> {
  const r = await fetch(path, {
    credentials: 'same-origin',
    ...(body !== undefined && {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),
  });
  if (r.status === 401) throw new AuthError();
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r;
}

/** 401 を false にする */
const authed = (p: Promise<unknown>) =>
  p.then(
    () => true,
    (e) => {
      if (e instanceof AuthError) return false;
      throw e;
    },
  );

const httpApi: Api = {
  login: (password) => authed(call('/api/login', { password })),
  logout: () => call('/api/logout', {}).then(() => {}),
  hasSession: () => authed(call('/api/session')),
  search: (embedding) => call('/api/search', { embedding: Array.from(embedding) }).then((r) => r.json()),
  thumbUrl: (p) => `/img/thumb/${p.id}`,
  largeUrl: (p) => `/img/large/${p.id}`,
  originalUrl: (p) => `/dl/${p.id}`,
};

export const api: Api = import.meta.env.VITE_MOCK === '1' ? mockApi : httpApi;
