import { clearCookie, isAuthed, issueCookie, passwordMatches } from './auth';
import { loadIndex, parseEmbedding, searchIndex } from './search';

interface Env {
  ASSETS: Fetcher;
  PHOTOS: R2Bucket;
  LOGIN_LIMITER: RateLimit;
  SITE_PASSWORD: string;
  SESSION_SECRET: string;
  SESSION_DAYS: string;
}

const json = (data: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(data), {
    ...init,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...init.headers },
  });
const status = (code: number, headers: HeadersInit = {}) =>
  new Response(null, { status: code, headers: { 'cache-control': 'no-store', ...headers } });

// R2 のキー: thumb/{id}.webp (長辺400) / large/{id}.webp (長辺1600) / orig/{id}.jpg (元画像)
const IMG_KINDS = {
  thumb: { key: (id: string) => `thumb/${id}.webp`, type: 'image/webp' },
  large: { key: (id: string) => `large/${id}.webp`, type: 'image/webp' },
} as const;
const ID_RE = /^[a-f0-9]{16}$/;

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!env.SESSION_SECRET || !env.SITE_PASSWORD) return new Response('server not configured', { status: 500 });
    const url = new URL(req.url);
    const path = url.pathname;
    const secure = url.protocol === 'https:';

    // ---- 認証なしで使えるもの
    if (path === '/api/login' && req.method === 'POST') {
      const ip = req.headers.get('cf-connecting-ip') ?? 'local';
      const { success } = await env.LOGIN_LIMITER.limit({ key: ip });
      if (!success) return json({ error: 'too_many_attempts' }, { status: 429 });
      const body = (await req.json().catch(() => null)) as { password?: unknown } | null;
      const pw = typeof body?.password === 'string' ? body.password.slice(0, 200) : '';
      if (!(await passwordMatches(pw, env.SITE_PASSWORD, env.SESSION_SECRET))) return status(401);
      return status(204, { 'set-cookie': await issueCookie(env.SESSION_SECRET, Number(env.SESSION_DAYS) || 7, secure) });
    }
    if (path === '/api/logout' && req.method === 'POST') {
      return status(204, { 'set-cookie': clearCookie(secure) });
    }

    // ---- ここから先はログイン必須
    if (!(await isAuthed(req, env.SESSION_SECRET))) return status(401);

    if (path === '/api/session' && req.method === 'GET') return status(204);

    if (path === '/api/search' && req.method === 'POST') {
      const index = await loadIndex(env.PHOTOS);
      const q = parseEmbedding(await req.json().catch(() => null), index.meta.dim);
      if (!q) return json({ error: 'bad_embedding' }, { status: 400 });
      const hits = searchIndex(index, q); // しきい値は索引 (meta.json) に入っている
      // 送られてきた特徴量はログにも残さない
      return json({
        total: index.meta.photos.length,
        photos: hits.map((h) => ({ ...h.photo, score: h.score, level: h.level })),
      });
    }

    const img = path.match(/^\/img\/(thumb|large)\/([^/]+)$/);
    if (img && req.method === 'GET') {
      const [, kind, id] = img as unknown as [string, keyof typeof IMG_KINDS, string];
      if (!ID_RE.test(id)) return status(404);
      return serveObject(env, req, IMG_KINDS[kind].key(id), IMG_KINDS[kind].type);
    }

    const dl = path.match(/^\/dl\/([^/]+)$/);
    if (dl && req.method === 'GET') {
      const id = dl[1];
      if (!ID_RE.test(id)) return status(404);
      const meta = (await loadIndex(env.PHOTOS)).byId.get(id);
      if (!meta) return status(404);
      const name = meta.name.replace(/[^\w.\-]/g, '_');
      return serveObject(env, req, `orig/${id}.jpg`, 'image/jpeg', {
        'content-disposition': `attachment; filename="${name}"; filename*=UTF-8''${encodeURIComponent(meta.name)}`,
      });
    }

    return status(404);
  },
} satisfies ExportedHandler<Env>;

/** R2 から読んでストリーミングで返す（ETag による 304 に対応） */
async function serveObject(env: Env, req: Request, key: string, type: string, extra: HeadersInit = {}) {
  const obj = await env.PHOTOS.get(key, { onlyIf: req.headers });
  if (!obj) return status(404);
  const headers = new Headers(extra);
  headers.set('content-type', type);
  headers.set('etag', obj.httpEtag);
  // 写真は変わらないのでブラウザに長くキャッシュさせる。共有キャッシュには載せない
  headers.set('cache-control', 'private, max-age=604800, immutable');
  if (!('body' in obj)) return new Response(null, { status: 304, headers });
  headers.set('content-length', String(obj.size));
  return new Response(obj.body, { headers });
}
