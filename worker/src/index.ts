import { clearCookie, isAuthed, issueCookie, passwordMatches } from './auth';
import { empty, json, type Ctx, type Env } from './http';
import { serveImage, serveOriginal } from './photos';
import { loadIndex, parseEmbedding, searchIndex } from './search';

interface Route {
  method: 'GET' | 'POST';
  pattern: RegExp;
  public?: boolean; // true ならログイン不要
  handle: (ctx: Ctx, params: string[]) => Response | Promise<Response>;
}

const ID = '([a-f0-9]{16})';

const routes: Route[] = [
  { method: 'POST', pattern: /^\/api\/login$/, public: true, handle: login },
  { method: 'POST', pattern: /^\/api\/logout$/, public: true, handle: (ctx) => empty(204, { 'set-cookie': clearCookie(ctx.secure) }) },
  { method: 'GET', pattern: /^\/api\/session$/, handle: () => empty(204) },
  { method: 'POST', pattern: /^\/api\/search$/, handle: search },
  { method: 'GET', pattern: new RegExp(`^/img/(thumb|large)/${ID}$`), handle: (ctx, [kind, id]) => serveImage(ctx, kind, id) },
  { method: 'GET', pattern: new RegExp(`^/dl/${ID}$`), handle: (ctx, [id]) => serveOriginal(ctx, id) },
];

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!env.SESSION_SECRET || !env.SITE_PASSWORD) return new Response('server not configured', { status: 500 });
    const url = new URL(req.url);
    const ctx: Ctx = { req, env, secure: url.protocol === 'https:' };

    let authed: boolean | undefined;
    for (const route of routes) {
      if (route.method !== req.method) continue;
      const m = url.pathname.match(route.pattern);
      if (!m) continue;
      if (!route.public && !(authed ??= await isAuthed(req, env.SESSION_SECRET))) return empty(401);
      return route.handle(ctx, m.slice(1));
    }
    // 存在しないパスでも、未ログインには 401 だけを返す
    return empty((await isAuthed(req, env.SESSION_SECRET)) ? 404 : 401);
  },
} satisfies ExportedHandler<Env>;

async function login({ req, env, secure }: Ctx) {
  const ip = req.headers.get('cf-connecting-ip') ?? 'local';
  if (!(await env.LOGIN_LIMITER.limit({ key: ip })).success) return json({ error: 'too_many_attempts' }, 429);

  const body = (await req.json().catch(() => null)) as { password?: unknown } | null;
  const password = typeof body?.password === 'string' ? body.password.slice(0, 200) : '';
  if (!(await passwordMatches(password, env.SITE_PASSWORD, env.SESSION_SECRET))) return empty(401);

  const days = Number(env.SESSION_DAYS) || 7;
  return empty(204, { 'set-cookie': await issueCookie(env.SESSION_SECRET, days, secure) });
}

/** 受け取った特徴量は保存もログ出力もしない */
async function search({ req, env }: Ctx) {
  const index = await loadIndex(env.PHOTOS);
  const query = parseEmbedding(await req.json().catch(() => null), index.meta.dim);
  if (!query) return json({ error: 'bad_embedding' }, 400);
  return json({
    total: index.meta.photos.length,
    photos: searchIndex(index, query).map((hit) => ({ ...hit.photo, score: hit.score, level: hit.level })),
  });
}
