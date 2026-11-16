export interface Env {
  ASSETS: Fetcher;
  PHOTOS: R2Bucket;
  LOGIN_LIMITER: RateLimit;
  SITE_PASSWORD: string;
  SESSION_SECRET: string;
  SESSION_DAYS: string;
}

export interface Ctx {
  req: Request;
  env: Env;
  secure: boolean; // https なら Cookie に Secure を付ける
}

export const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });

export const empty = (status: number, headers: HeadersInit = {}) =>
  new Response(null, { status, headers: { 'cache-control': 'no-store', ...headers } });
