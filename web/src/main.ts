import './style.css';
import { downloadZip } from 'client-zip';
import { AuthError, hasSession, login, logout, search, urls, type Photo } from './api';
import { NoFaceError, extractFace, loadImage, loadModels } from './face';

// ------------------------------------------------------------ helpers
type Child = Node | string | null | undefined | false;
function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, unknown> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'class') el.className = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c) el.append(c);
  return el;
}
const mb = (b: number) => (b / 1e6 >= 1000 ? `${(b / 1e9).toFixed(1)}GB` : `${Math.round(b / 1e6)}MB`);
const main = document.getElementById('main')!;
const logoutBtn = document.getElementById('logout') as HTMLButtonElement;

type Step = 'login' | 'upload' | 'results';
function show(step: Step, ...nodes: Node[]) {
  document.querySelectorAll<HTMLElement>('.steps [data-step]').forEach((s) => {
    s.classList.toggle('on', s.dataset.step === step);
  });
  logoutBtn.hidden = step === 'login';
  main.replaceChildren(...nodes);
  window.scrollTo({ top: 0 });
}

function onAuthLost(e: unknown) {
  if (e instanceof AuthError) {
    renderLogin('セッションが切れました。もう一度パスワードを入力してください。');
    return true;
  }
  return false;
}

// ------------------------------------------------------------ 01 login
function renderLogin(message = '') {
  const input = h('input', {
    type: 'password',
    name: 'password',
    autocomplete: 'current-password',
    placeholder: 'パスワード',
    required: true,
    'aria-label': 'パスワード',
  });
  const err = h('p', { class: 'form-error', role: 'alert' }, message);
  const btn = h('button', { class: 'pill pill-dark', type: 'submit' }, '入る', h('span', { class: 'arrow' }, '→'));
  const form = h(
    'form',
    {
      class: 'login-form',
      onsubmit: async (e: Event) => {
        e.preventDefault();
        btn.disabled = true;
        err.textContent = '';
        try {
          if (await login(input.value)) renderUpload();
          else {
            err.textContent = 'パスワードが違います。';
            input.select();
          }
        } catch {
          err.textContent = '接続できませんでした。時間をおいて試してください。';
        } finally {
          btn.disabled = false;
        }
      },
    },
    input,
    btn,
  );

  show(
    'login',
    h(
      'section',
      { class: 'hero dots' },
      h('p', { class: 'eyebrow' }, 'HACKATHON PHOTO FINDER'),
      h('h1', {}, 'Face Finder'),
      h(
        'p',
        { class: 'lead' },
        'ハッカソンで撮った写真の中から、顔写真 1 枚であなたが写っているものを集めます。元の解像度のままダウンロードできます。',
      ),
      form,
      err,
      h('p', { class: 'hint' }, 'パスワードは配布されたものを使ってください。'),
    ),
  );
  input.focus();
}

// ------------------------------------------------------------ 02 upload
function renderUpload() {
  loadModels().catch(() => {}); // 写真を選んでいる間にモデルを先読み
  const status = h('div', { class: 'status', role: 'status', 'aria-live': 'polite' });
  const fileInput = h('input', { type: 'file', accept: 'image/*', hidden: true });
  const camInput = h('input', { type: 'file', accept: 'image/*', capture: 'user', hidden: true });

  async function handle(file: File | undefined) {
    if (!file) return;
    drop.classList.add('busy');
    const say = (s: string) => status.replaceChildren(h('span', { class: 'spinner' }), s);
    say('写真を読み込んでいます…');
    try {
      const img = await loadImage(file).catch(() => {
        throw new Error('この形式の画像は読めませんでした。JPEG か PNG で試してください。');
      });
      const face = await extractFace(img, say);
      status.replaceChildren(
        h('img', { class: 'face-mini', src: face.preview, alt: '' }),
        face.faceCount > 1 ? `顔が ${face.faceCount} 人写っていたので、一番大きい顔で探しています…` : '写真を探しています…',
      );
      const res = await search(face.embedding);
      renderResults(res.photos, res.total, face.preview);
    } catch (e) {
      if (onAuthLost(e)) return;
      const msg =
        e instanceof NoFaceError
          ? '顔が見つかりませんでした。顔が大きく、正面から写った写真で試してください。'
          : e instanceof Error && e.message.startsWith('この形式')
            ? e.message
            : 'うまくいきませんでした。もう一度試してください。';
      status.replaceChildren(h('span', { class: 'status-error' }, msg));
    } finally {
      drop.classList.remove('busy');
      fileInput.value = camInput.value = '';
    }
  }
  fileInput.addEventListener('change', () => handle(fileInput.files?.[0]));
  camInput.addEventListener('change', () => handle(camInput.files?.[0]));

  const drop = h(
    'div',
    {
      class: 'drop',
      ondragover: (e: DragEvent) => {
        e.preventDefault();
        drop.classList.add('over');
      },
      ondragleave: () => drop.classList.remove('over'),
      ondrop: (e: DragEvent) => {
        e.preventDefault();
        drop.classList.remove('over');
        handle(e.dataTransfer?.files[0]);
      },
    },
    h('span', { class: 'drop-ring', 'aria-hidden': 'true' }),
    h('p', { class: 'drop-title' }, '顔写真をここにドロップ'),
    h(
      'div',
      { class: 'drop-actions' },
      h('button', { class: 'pill pill-dark', type: 'button', onclick: () => fileInput.click() }, '写真を選ぶ'),
      h('button', { class: 'pill pill-line', type: 'button', onclick: () => camInput.click() }, 'カメラで撮る'),
    ),
    fileInput,
    camInput,
    status,
  );

  show(
    'upload',
    h(
      'section',
      { class: 'section' },
      h('p', { class: 'section-top' }, h('b', {}, '02'), ' — YOUR FACE'),
      h(
        'div',
        { class: 'split-card' },
        h('div', { class: 'split-main' }, drop),
        h(
          'div',
          { class: 'split-side tone-green' },
          h('p', { class: 'tag' }, 'Tips'),
          h(
            'ul',
            { class: 'tips' },
            h('li', {}, '正面から顔が大きく写った写真'),
            h('li', {}, '明るい場所で撮影（マスク・サングラスなし）'),
            h('li', {}, '1人で写っている写真（自撮りもOK）'),
          ),
          h('p', { class: 'tag' }, 'Privacy'),
          h(
            'p',
            { class: 'side-note' },
            '顔写真はこの端末の中で数値（特徴量）に変換され、写真そのものは送信も保存もされません。',
          ),
        ),
      ),
    ),
  );
}

// ------------------------------------------------------------ 03 results
function renderResults(photos: Photo[], total: number, facePreview: string) {
  const hidden = new Set<string>();
  let filter: 'all' | 'sure' | 'maybe' = 'all';
  const visible = () =>
    photos.filter((p) => !hidden.has(p.id) && (filter === 'all' || p.level === filter));
  const kept = () => photos.filter((p) => !hidden.has(p.id));

  const title = h('h2', {});
  const sub = h('p', { class: 'lead' });
  const zipBtn = h('button', { class: 'pill pill-dark', type: 'button', onclick: () => downloadAll() });
  const zipStatus = h('p', { class: 'zip-status', role: 'status', 'aria-live': 'polite' });
  const tabs = h('div', { class: 'tabs', role: 'tablist' });
  const hiddenBar = h('div', { class: 'hidden-bar' });
  const grid = h('div', { class: 'grid' });

  function refresh() {
    const k = kept();
    const sure = k.filter((p) => p.level === 'sure').length;
    title.textContent = k.length ? `${k.length} 枚見つかりました。` : '見つかりませんでした。';
    // 見つからなかったときだけ案内を出す
    sub.textContent = k.length ? '' : '別の写真（正面・明るいもの）で試してみてください。';
    sub.hidden = k.length > 0;
    zipBtn.replaceChildren(
      `まとめてダウンロード（${k.length} 枚・${mb(k.reduce((s, p) => s + p.bytes, 0))}）`,
      h('span', { class: 'arrow' }, '↓'),
    );
    zipBtn.disabled = k.length === 0;

    tabs.replaceChildren(
      ...(
        [
          ['all', 'すべて', k.length],
          ['sure', 'ほぼ確実', sure],
          ['maybe', 'たぶん本人', k.length - sure],
        ] as const
      ).map(([key, label, n]) =>
        h(
          'button',
          {
            class: `tab${filter === key ? ' on' : ''}`,
            role: 'tab',
            'aria-selected': filter === key ? 'true' : 'false',
            type: 'button',
            onclick: () => {
              filter = key;
              refresh();
            },
          },
          label,
          h('sup', {}, String(n)),
        ),
      ),
    );

    hiddenBar.replaceChildren(
      hidden.size
        ? h(
            'p',
            {},
            `「自分じゃない」にした写真 ${hidden.size} 枚を隠しています。`,
            h(
              'button',
              {
                class: 'link',
                type: 'button',
                onclick: () => {
                  hidden.clear();
                  refresh();
                },
              },
              '元に戻す',
            ),
          )
        : '',
    );

    const list = visible();
    grid.replaceChildren(
      ...list.map((p, i) =>
        h(
          'figure',
          { class: 'shot' },
          h(
            'button',
            { class: 'shot-open', type: 'button', onclick: () => openViewer(list, i), 'aria-label': `${p.name} を拡大` },
            h('img', {
              src: urls.thumb(p),
              alt: '',
              loading: 'lazy',
              width: p.width > p.height ? 400 : 267,
              height: p.width > p.height ? 267 : 400,
            }),
          ),
          h(
            'figcaption',
            {},
            h('span', { class: `badge ${p.level}` }, p.level === 'sure' ? 'ほぼ確実' : 'たぶん'),
            h('span', { class: 'shot-name' }, p.name),
            h(
              'button',
              {
                class: 'not-me',
                type: 'button',
                title: '自分じゃない',
                'aria-label': `${p.name} は自分じゃない`,
                onclick: () => {
                  hidden.add(p.id);
                  refresh();
                },
              },
              '×',
            ),
          ),
        ),
      ),
    );
  }

  // まとめてダウンロード: 元画像を順に取得して無圧縮 ZIP をストリーミング生成
  async function downloadAll() {
    const list = kept();
    zipBtn.disabled = true;
    let done = 0;
    async function* files() {
      for (const p of list) {
        const r = await fetch(urls.original(p), { credentials: 'same-origin' });
        if (r.status === 401) throw new AuthError();
        if (!r.ok) throw new Error(p.name);
        done++;
        zipStatus.textContent = `${done} / ${list.length} 枚を取得中…`;
        yield { name: p.name, input: r };
      }
    }
    const filename = 'hackathon-photos.zip';
    try {
      const picker = (window as unknown as {
        showSaveFilePicker?: (o: object) => Promise<FileSystemFileHandle>;
      }).showSaveFilePicker;
      if (picker) {
        // 対応ブラウザではファイルに直接書き込む（メモリに全部載せない）
        const handle = await picker({ suggestedName: filename, types: [{ accept: { 'application/zip': ['.zip'] } }] });
        await downloadZip(files()).body!.pipeTo(await handle.createWritable());
      } else {
        const blob = await downloadZip(files()).blob();
        const a = h('a', { href: URL.createObjectURL(blob), download: filename });
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
      }
      zipStatus.textContent = `${list.length} 枚を ZIP で保存しました。`;
    } catch (e) {
      if (onAuthLost(e)) return;
      if (e instanceof DOMException && e.name === 'AbortError') zipStatus.textContent = '';
      else zipStatus.textContent = `途中で失敗しました（${done} 枚目まで取得）。もう一度試してください。`;
    } finally {
      zipBtn.disabled = false;
    }
  }

  show(
    'results',
    h(
      'section',
      { class: 'section' },
      h('p', { class: 'section-top' }, h('b', {}, '03'), ' — YOUR PHOTOS'),
      h(
        'div',
        { class: 'result-head' },
        h('img', { class: 'face-big', src: facePreview, alt: '検索に使った顔' }),
        h('div', {}, title, sub),
      ),
      h(
        'div',
        { class: 'actions' },
        zipBtn,
        h(
          'button',
          { class: 'pill pill-line', type: 'button', onclick: () => renderUpload() },
          '別の顔で探す',
          h('span', { class: 'arrow' }, '→'),
        ),
      ),
      zipStatus,
      h('div', { class: 'toolbar' }, tabs, hiddenBar),
      grid,
    ),
  );
  refresh();

  // ---------------------------------------------------------- viewer
  function openViewer(list: Photo[], start: number) {
    let i = start;
    const img = h('img', { class: 'viewer-img', alt: '' });
    const meta = h('p', { class: 'viewer-meta' });
    const dl = h('a', { class: 'pill pill-paper' });
    const share = h('button', { class: 'pill pill-line-paper', type: 'button', hidden: true }, '保存・共有');
    const counter = h('span', { class: 'viewer-count' });

    function go(n: number) {
      i = (n + list.length) % list.length;
      const p = list[i];
      img.src = urls.large(p);
      meta.replaceChildren(
        h('span', { class: `badge ${p.level}` }, p.level === 'sure' ? 'ほぼ確実' : 'たぶん'),
        ` ${p.name} ・ ${p.width}×${p.height} ・ 類似度 ${p.score.toFixed(2)}`,
      );
      dl.href = urls.original(p);
      dl.setAttribute('download', p.name);
      dl.replaceChildren(`元画像をダウンロード（${mb(p.bytes)}）`, h('span', { class: 'arrow' }, '↓'));
      counter.textContent = `${i + 1} / ${list.length}`;
    }

    // スマホ: 共有シートから「写真に保存」できるように
    if ('canShare' in navigator) {
      share.hidden = false;
      share.onclick = async () => {
        const p = list[i];
        try {
          const blob = await (await fetch(urls.original(p), { credentials: 'same-origin' })).blob();
          const file = new File([blob], p.name, { type: 'image/jpeg' });
          if (navigator.canShare({ files: [file] })) await navigator.share({ files: [file] });
          else dl.click();
        } catch {
          /* キャンセル */
        }
      };
    }

    const dlg = h(
      'dialog',
      { class: 'viewer', 'aria-label': '写真を拡大' },
      h(
        'div',
        { class: 'viewer-top' },
        counter,
        h('button', { class: 'viewer-close', type: 'button', 'aria-label': '閉じる', onclick: () => dlg.close() }, '×'),
      ),
      h(
        'div',
        { class: 'viewer-stage' },
        h('button', { class: 'nav prev', type: 'button', 'aria-label': '前の写真', onclick: () => go(i - 1) }, '←'),
        img,
        h('button', { class: 'nav next', type: 'button', 'aria-label': '次の写真', onclick: () => go(i + 1) }, '→'),
      ),
      h(
        'div',
        { class: 'viewer-bottom' },
        meta,
        h(
          'div',
          { class: 'viewer-actions' },
          dl,
          share,
          h(
            'button',
            {
              class: 'pill pill-line-paper',
              type: 'button',
              onclick: () => {
                hidden.add(list[i].id);
                dlg.close();
                refresh();
              },
            },
            '自分じゃない',
          ),
        ),
      ),
    );
    dlg.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') go(i - 1);
      if (e.key === 'ArrowRight') go(i + 1);
    });
    // スワイプで前後
    let x0: number | null = null;
    img.addEventListener('touchstart', (e) => (x0 = e.touches[0].clientX), { passive: true });
    img.addEventListener('touchend', (e) => {
      if (x0 == null) return;
      const dx = e.changedTouches[0].clientX - x0;
      if (Math.abs(dx) > 50) go(dx > 0 ? i - 1 : i + 1);
      x0 = null;
    });
    dlg.addEventListener('close', () => dlg.remove());
    dlg.addEventListener('click', (e) => e.target === dlg && dlg.close());
    document.body.append(dlg);
    go(i);
    dlg.showModal();
  }
}

// ------------------------------------------------------------ boot
logoutBtn.addEventListener('click', async () => {
  await logout().catch(() => {});
  renderLogin();
});

hasSession()
  .then((ok) => (ok ? renderUpload() : renderLogin()))
  .catch(() => renderLogin('接続できませんでした。時間をおいて再読み込みしてください。'));
