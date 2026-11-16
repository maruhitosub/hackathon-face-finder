import type { Level } from './api';

type Child = Node | string | null | undefined | false;

/** 要素を作る。on* は イベント、class は className、それ以外は属性（false / null は付けない） */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, unknown> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value as EventListener);
    else if (key === 'class') el.className = String(value);
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children) if (child) el.append(child);
  return el;
}

export const button = (cls: string, label: Child, onclick: () => void, ...rest: Child[]) =>
  h('button', { class: cls, type: 'button', onclick }, label, ...rest);

export const arrow = (char: string) => h('span', { class: 'arrow' }, char);

export const badge = (level: Level) =>
  h('span', { class: `badge ${level}` }, level === 'sure' ? 'ほぼ確実' : 'たぶん');

export const formatBytes = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)}GB` : `${Math.round(b / 1e6)}MB`);
