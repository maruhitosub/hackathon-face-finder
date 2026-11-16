import { api, AuthError, type SearchResponse } from '../api';
import { button, h } from '../dom';
import { extractFace, loadImage, loadModels, NoFaceError } from '../face';

class UnreadableImageError extends Error {}

function errorMessage(e: unknown): string {
  if (e instanceof NoFaceError) return '顔が見つかりませんでした。顔が大きく、正面から写った写真で試してください。';
  if (e instanceof UnreadableImageError) return 'この形式の画像は読めませんでした。JPEG か PNG で試してください。';
  return 'うまくいきませんでした。もう一度試してください。';
}

export function uploadView(opts: {
  onResult: (res: SearchResponse, facePreview: string) => void;
  onAuthLost: () => void;
}): HTMLElement {
  loadModels().catch(() => {}); // 写真を選んでいる間に先読み

  const status = h('div', { class: 'status', role: 'status', 'aria-live': 'polite' });
  const say = (text: string) => status.replaceChildren(h('span', { class: 'spinner' }), text);
  const fileInput = h('input', { type: 'file', accept: 'image/*', hidden: true });
  const cameraInput = h('input', { type: 'file', accept: 'image/*', capture: 'user', hidden: true });

  async function handle(file: File | undefined) {
    if (!file) return;
    drop.classList.add('busy');
    say('写真を読み込んでいます…');
    try {
      const image = await loadImage(file).catch(() => {
        throw new UnreadableImageError();
      });
      const face = await extractFace(image, say);
      status.replaceChildren(
        h('img', { class: 'face-mini', src: face.preview, alt: '' }),
        face.faceCount > 1 ? `顔が ${face.faceCount} 人写っていたので、一番大きい顔で探しています…` : '写真を探しています…',
      );
      opts.onResult(await api.search(face.embedding), face.preview);
    } catch (e) {
      if (e instanceof AuthError) return opts.onAuthLost();
      status.replaceChildren(h('span', { class: 'status-error' }, errorMessage(e)));
    } finally {
      drop.classList.remove('busy');
      fileInput.value = cameraInput.value = '';
    }
  }
  fileInput.addEventListener('change', () => handle(fileInput.files?.[0]));
  cameraInput.addEventListener('change', () => handle(cameraInput.files?.[0]));

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
      button('pill pill-dark', '写真を選ぶ', () => fileInput.click()),
      button('pill pill-line', 'カメラで撮る', () => cameraInput.click()),
    ),
    fileInput,
    cameraInput,
    status,
  );

  const side = h(
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
    h('p', { class: 'side-note' }, '顔写真はこの端末の中で数値（特徴量）に変換され、写真そのものは送信も保存もされません。'),
  );

  return h(
    'section',
    { class: 'section' },
    h('p', { class: 'section-top' }, h('b', {}, '02'), ' — YOUR FACE'),
    h('div', { class: 'split-card' }, h('div', { class: 'split-main' }, drop), side),
  );
}
