import { api, type Photo } from '../api';
import { arrow, badge, button, formatBytes, h } from '../dom';

const SWIPE_PX = 50;

/** 全画面の拡大表示。←→ キー・スワイプで前後に移動 */
export function openViewer(photos: Photo[], start: number, onNotMe: (p: Photo) => void) {
  let index = start;
  const image = h('img', { class: 'viewer-img', alt: '' });
  const meta = h('p', { class: 'viewer-meta' });
  const download = h('a', { class: 'pill pill-paper' });
  const counter = h('span', { class: 'viewer-count' });

  function go(to: number) {
    index = (to + photos.length) % photos.length;
    const p = photos[index];
    image.src = api.largeUrl(p);
    meta.replaceChildren(badge(p.level), ` ${p.name} ・ ${p.width}×${p.height} ・ 類似度 ${p.score.toFixed(2)}`);
    download.href = api.originalUrl(p);
    download.download = p.name;
    download.replaceChildren(`元画像をダウンロード（${formatBytes(p.bytes)}）`, arrow('↓'));
    counter.textContent = `${index + 1} / ${photos.length}`;
  }

  /** スマホでは共有シートから写真アプリに保存できる */
  async function share() {
    const p = photos[index];
    try {
      const blob = await (await fetch(api.originalUrl(p), { credentials: 'same-origin' })).blob();
      const file = new File([blob], p.name, { type: 'image/jpeg' });
      if (navigator.canShare({ files: [file] })) await navigator.share({ files: [file] });
      else download.click();
    } catch {
      // 共有をキャンセルした
    }
  }

  const dialog = h(
    'dialog',
    { class: 'viewer', 'aria-label': '写真を拡大' },
    h(
      'div',
      { class: 'viewer-top' },
      counter,
      h('button', { class: 'viewer-close', type: 'button', 'aria-label': '閉じる', onclick: () => dialog.close() }, '×'),
    ),
    h(
      'div',
      { class: 'viewer-stage' },
      h('button', { class: 'nav prev', type: 'button', 'aria-label': '前の写真', onclick: () => go(index - 1) }, '←'),
      image,
      h('button', { class: 'nav next', type: 'button', 'aria-label': '次の写真', onclick: () => go(index + 1) }, '→'),
    ),
    h(
      'div',
      { class: 'viewer-bottom' },
      meta,
      h(
        'div',
        { class: 'viewer-actions' },
        download,
        'canShare' in navigator && button('pill pill-line-paper', '保存・共有', share),
        button('pill pill-line-paper', '自分じゃない', () => {
          dialog.close();
          onNotMe(photos[index]);
        }),
      ),
    ),
  );

  dialog.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') go(index - 1);
    if (e.key === 'ArrowRight') go(index + 1);
  });
  let touchX: number | null = null;
  image.addEventListener('touchstart', (e) => (touchX = e.touches[0].clientX), { passive: true });
  image.addEventListener('touchend', (e) => {
    if (touchX == null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > SWIPE_PX) go(dx > 0 ? index - 1 : index + 1);
    touchX = null;
  });
  dialog.addEventListener('close', () => dialog.remove());
  dialog.addEventListener('click', (e) => e.target === dialog && dialog.close());

  document.body.append(dialog);
  go(index);
  dialog.showModal();
}
