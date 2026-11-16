// 共通パスワードでログイン → HMAC 署名付き Cookie。
// Cookie の中身は「有効期限.署名」だけで、サーバー側にセッションは持たない。

const COOKIE = 'hff_session';
const enc = new TextEncoder();

async function hmacKey(secret: string) {
  return crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, [
    'sign',
    'verify',
  ]);
}

const b64url = (buf: ArrayBuffer) =>
  btoa(String.fromCharCode(...new Uint8Array(buf)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

function fromB64url(s: string): Uint8Array {
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

/** パスワード比較（長さや一致位置で時間が変わらないよう、両方を HMAC してから比べる） */
export async function passwordMatches(input: string, expected: string, secret: string): Promise<boolean> {
  const key = await hmacKey(secret);
  const [a, b] = await Promise.all([
    crypto.subtle.sign('HMAC', key, enc.encode(`pw:${input}`)),
    crypto.subtle.sign('HMAC', key, enc.encode(`pw:${expected}`)),
  ]);
  return crypto.subtle.timingSafeEqual(a, b);
}

export async function issueCookie(secret: string, days: number, secure: boolean): Promise<string> {
  const exp = Math.floor(Date.now() / 1000) + days * 86400;
  const sig = await crypto.subtle.sign('HMAC', await hmacKey(secret), enc.encode(`s:${exp}`));
  const attrs = [`${COOKIE}=${exp}.${b64url(sig)}`, 'Path=/', 'HttpOnly', 'SameSite=Lax', `Max-Age=${days * 86400}`];
  if (secure) attrs.push('Secure');
  return attrs.join('; ');
}

export function clearCookie(secure: boolean): string {
  return `${COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0${secure ? '; Secure' : ''}`;
}

export async function isAuthed(req: Request, secret: string): Promise<boolean> {
  const raw = req.headers.get('cookie') ?? '';
  const m = raw.match(new RegExp(`(?:^|;\\s*)${COOKIE}=(\\d+)\\.([A-Za-z0-9_-]+)`));
  if (!m) return false;
  const exp = Number(m[1]);
  if (!(exp > Date.now() / 1000)) return false;
  try {
    return await crypto.subtle.verify('HMAC', await hmacKey(secret), fromB64url(m[2]), enc.encode(`s:${exp}`));
  } catch {
    return false;
  }
}
