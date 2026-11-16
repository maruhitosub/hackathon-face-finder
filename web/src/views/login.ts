import { api } from '../api';
import { arrow, h } from '../dom';

export function loginView(opts: { message?: string; onLogin: () => void }): HTMLElement {
  const input = h('input', {
    type: 'password',
    name: 'password',
    autocomplete: 'current-password',
    placeholder: 'パスワード',
    required: true,
    autofocus: true,
    'aria-label': 'パスワード',
  });
  const error = h('p', { class: 'form-error', role: 'alert' }, opts.message);
  const submit = h('button', { class: 'pill pill-dark', type: 'submit' }, '入る', arrow('→'));

  async function onsubmit(e: Event) {
    e.preventDefault();
    submit.disabled = true;
    error.textContent = '';
    try {
      if (await api.login(input.value)) return opts.onLogin();
      error.textContent = 'パスワードが違います。';
      input.select();
    } catch {
      error.textContent = '接続できませんでした。時間をおいて試してください。';
    } finally {
      submit.disabled = false;
    }
  }

  return h(
    'section',
    { class: 'hero dots' },
    h('p', { class: 'eyebrow' }, 'HACKATHON PHOTO FINDER'),
    h('h1', {}, 'Face Finder'),
    h(
      'p',
      { class: 'lead' },
      'ハッカソンで撮った写真の中から、顔写真 1 枚であなたが写っているものを集めます。元の解像度のままダウンロードできます。',
    ),
    h('form', { class: 'login-form', onsubmit }, input, submit),
    error,
    h('p', { class: 'hint' }, 'パスワードは配布されたものを使ってください。'),
  );
}
