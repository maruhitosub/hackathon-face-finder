// 共通パスワード → HMAC 署名付き Cookie「有効期限.署名」。サーバー側にセッションは持たない。

const COOKIE = 'hff_session';
const COOKIE_RE = new RegExp(`(?:^|;\\s*)${COOKIE}=(\\d+)\\.([A-Za-z0-9_-]+)`);
const enc = new TextEncoder();

const hmacKey = (secret: string) =>
  crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);

const toBase64Url = (buf: ArrayBuffer) =>
  btoa(String.fromCharCode(...new Uint8Array(buf)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

const fromBase64Url = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0));

const cookieAttrs = (maxAge: number, secure: boolean) =>
  `Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAge}${secure ? '; Secure' : ''}`;

/** 両方を HMAC してから比べる（長さや一致位置で処理時間が変わらない） */
export async function passwordMatches(input: string, expected: string, secret: string): Promise<boolean> {
  const key = await hmacKey(secret);
  const [a, b] = await Promise.all(
    [input, expected].map((pw) => crypto.subtle.sign('HMAC', key, enc.encode(`pw:${pw}`))),
  );
  return crypto.subtle.timingSafeEqual(a, b);
}

export async function issueCookie(secret: string, days: number, secure: boolean): Promise<string> {
  const maxAge = days * 86400;
  const exp = Math.floor(Date.now() / 1000) + maxAge;
  const sig = await crypto.subtle.sign('HMAC', await hmacKey(secret), enc.encode(`s:${exp}`));
  return `${COOKIE}=${exp}.${toBase64Url(sig)}; ${cookieAttrs(maxAge, secure)}`;
}

export const clearCookie = (secure: boolean) => `${COOKIE}=; ${cookieAttrs(0, secure)}`;

export async function isAuthed(req: Request, secret: string): Promise<boolean> {
  const m = (req.headers.get('cookie') ?? '').match(COOKIE_RE);
  if (!m) return false;
  const exp = Number(m[1]);
  if (!(exp > Date.now() / 1000)) return false;
  try {
    return await crypto.subtle.verify('HMAC', await hmacKey(secret), fromBase64Url(m[2]), enc.encode(`s:${exp}`));
  } catch {
    return false;
  }
}
