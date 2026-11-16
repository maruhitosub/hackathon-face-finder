import './style.css';
import { api, type SearchResponse } from './api';
import { loginView } from './views/login';
import { resultsView } from './views/results';
import { uploadView } from './views/upload';

type Step = 'login' | 'upload' | 'results';

const main = document.getElementById('main')!;
const logoutButton = document.getElementById('logout') as HTMLButtonElement;

function show(step: Step, view: HTMLElement) {
  for (const el of document.querySelectorAll<HTMLElement>('.steps [data-step]')) {
    el.classList.toggle('on', el.dataset.step === step);
  }
  logoutButton.hidden = step === 'login';
  main.replaceChildren(view);
  window.scrollTo({ top: 0 });
  view.querySelector<HTMLElement>('[autofocus]')?.focus();
}

const showLogin = (message?: string) => show('login', loginView({ message, onLogin: showUpload }));
const onAuthLost = () => showLogin('セッションが切れました。もう一度パスワードを入力してください。');

function showUpload() {
  show('upload', uploadView({ onResult: showResults, onAuthLost }));
}

function showResults(res: SearchResponse, facePreview: string) {
  show('results', resultsView({ photos: res.photos, facePreview, onRetry: showUpload, onAuthLost }));
}

logoutButton.addEventListener('click', async () => {
  await api.logout().catch(() => {});
  showLogin();
});

api
  .hasSession()
  .then((ok) => (ok ? showUpload() : showLogin()))
  .catch(() => showLogin('接続できませんでした。時間をおいて再読み込みしてください。'));
