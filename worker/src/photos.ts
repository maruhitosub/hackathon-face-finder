// R2 の写真を返す。キー: thumb/{id}.webp (400px) / large/{id}.webp (1600px) / orig/{id}.jpg (元画像)

import { empty, type Ctx } from './http';
import { loadIndex } from './search';

export function serveImage(ctx: Ctx, kind: string, id: string) {
  return serveObject(ctx, `${kind}/${id}.webp`, 'image/webp');
}

export async function serveOriginal(ctx: Ctx, id: string) {
  const photo = (await loadIndex(ctx.env.PHOTOS)).byId.get(id);
  if (!photo) return empty(404);
  const ascii = photo.name.replace(/[^\w.\-]/g, '_');
  return serveObject(ctx, `orig/${id}.jpg`, 'image/jpeg', {
    'content-disposition': `attachment; filename="${ascii}"; filename*=UTF-8''${encodeURIComponent(photo.name)}`,
  });
}

/** ストリーミングで返す。If-None-Match には 304 */
async function serveObject({ req, env }: Ctx, key: string, type: string, extra: HeadersInit = {}) {
  const obj = await env.PHOTOS.get(key, { onlyIf: req.headers });
  if (!obj) return empty(404);
  const headers = new Headers(extra);
  headers.set('content-type', type);
  headers.set('etag', obj.httpEtag);
  headers.set('cache-control', 'private, max-age=604800, immutable'); // 共有キャッシュには載せない
  if (!('body' in obj)) return new Response(null, { status: 304, headers });
  headers.set('content-length', String(obj.size));
  return new Response(obj.body, { headers });
}
