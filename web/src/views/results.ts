import { api, AuthError, type Level, type Photo } from '../api';
import { arrow, badge, button, formatBytes, h } from '../dom';
import { saveAsZip } from '../zip';
import { openViewer } from './viewer';

type Filter = 'all' | Level;

const TABS: [Filter, string][] = [
  ['all', 'すべて'],
  ['sure', 'ほぼ確実'],
  ['maybe', 'たぶん本人'],
];

export function resultsView(opts: {
  photos: Photo[];
  facePreview: string;
  onRetry: () => void;
  onAuthLost: () => void;
}): HTMLElement {
  const hidden = new Set<string>(); // 「自分じゃない」にした写真
  let filter: Filter = 'all';
  const kept = () => opts.photos.filter((p) => !hidden.has(p.id));
  const matches = (p: Photo, f: Filter) => f === 'all' || p.level === f;

  const title = h('h2', {});
  const emptyHint = h('p', { class: 'lead', hidden: true }, '別の写真（正面・明るいもの）で試してみてください。');
  const zipButton = button('pill pill-dark', '', downloadAll);
  const zipStatus = h('p', { class: 'zip-status', role: 'status', 'aria-live': 'polite' });
  const tabs = h('div', { class: 'tabs', role: 'tablist' });
  const hiddenBar = h('div', { class: 'hidden-bar' });
  const grid = h('div', { class: 'grid' });

  function hide(p: Photo) {
    hidden.add(p.id);
    refresh();
  }

  function card(p: Photo, i: number, list: Photo[]) {
    const landscape = p.width > p.height;
    return h(
      'figure',
      { class: 'shot' },
      h(
        'button',
        { class: 'shot-open', type: 'button', 'aria-label': `${p.name} を拡大`, onclick: () => openViewer(list, i, hide) },
        h('img', {
          src: api.thumbUrl(p),
          alt: '',
          loading: 'lazy',
          width: landscape ? 400 : 267,
          height: landscape ? 267 : 400,
        }),
      ),
      h(
        'figcaption',
        {},
        badge(p.level),
        h('span', { class: 'shot-name' }, p.name),
        h(
          'button',
          { class: 'not-me', type: 'button', title: '自分じゃない', 'aria-label': `${p.name} は自分じゃない`, onclick: () => hide(p) },
          '×',
        ),
      ),
    );
  }

  function refresh() {
    const photos = kept();
    title.textContent = photos.length ? `${photos.length} 枚見つかりました。` : '見つかりませんでした。';
    emptyHint.hidden = photos.length > 0;

    const bytes = photos.reduce((sum, p) => sum + p.bytes, 0);
    zipButton.replaceChildren(`まとめてダウンロード（${photos.length} 枚・${formatBytes(bytes)}）`, arrow('↓'));
    zipButton.disabled = photos.length === 0;

    tabs.replaceChildren(
      ...TABS.map(([key, label]) =>
        h(
          'button',
          {
            class: `tab${filter === key ? ' on' : ''}`,
            role: 'tab',
            'aria-selected': String(filter === key),
            type: 'button',
            onclick: () => {
              filter = key;
              refresh();
            },
          },
          label,
          h('sup', {}, String(photos.filter((p) => matches(p, key)).length)),
        ),
      ),
    );

    hiddenBar.replaceChildren(
      hidden.size > 0
        ? h(
          'p',
          {},
          `「自分じゃない」にした写真 ${hidden.size} 枚を隠しています。`,
          button('link', '元に戻す', () => {
            hidden.clear();
            refresh();
          }),
          )
        : '',
    );

    grid.replaceChildren(...photos.filter((p) => matches(p, filter)).map(card));
  }

  async function downloadAll() {
    const photos = kept();
    let done = 0;
    zipButton.disabled = true;
    try {
      const saved = await saveAsZip(photos, 'hackathon-photos.zip', (n) => {
        done = n;
        zipStatus.textContent = `${n} / ${photos.length} 枚を取得中…`;
      });
      zipStatus.textContent = saved ? `${photos.length} 枚を ZIP で保存しました。` : '';
    } catch (e) {
      if (e instanceof AuthError) return opts.onAuthLost();
      zipStatus.textContent = `途中で失敗しました（${done} 枚目まで取得）。もう一度試してください。`;
    } finally {
      zipButton.disabled = false;
    }
  }

  refresh();
  return h(
    'section',
    { class: 'section' },
    h('p', { class: 'section-top' }, h('b', {}, '03'), ' — YOUR PHOTOS'),
    h(
      'div',
      { class: 'result-head' },
      h('img', { class: 'face-big', src: opts.facePreview, alt: '検索に使った顔' }),
      h('div', {}, title, emptyHint),
    ),
    h('div', { class: 'actions' }, zipButton, button('pill pill-line', '別の顔で探す', opts.onRetry, arrow('→'))),
    zipStatus,
    h('div', { class: 'toolbar' }, tabs, hiddenBar),
    grid,
  );
}
