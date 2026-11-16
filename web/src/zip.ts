import { downloadZip } from 'client-zip';
import { api, AuthError, type Photo } from './api';

type SavePicker = (options: object) => Promise<FileSystemFileHandle>;

/** 元画像を順に取得しながら無圧縮 ZIP にして保存する。保存をキャンセルしたら false */
export async function saveAsZip(photos: Photo[], filename: string, onProgress: (done: number) => void): Promise<boolean> {
  let done = 0;
  async function* files() {
    for (const p of photos) {
      const r = await fetch(api.originalUrl(p), { credentials: 'same-origin' });
      if (r.status === 401) throw new AuthError();
      if (!r.ok) throw new Error(p.name);
      onProgress(++done);
      yield { name: p.name, input: r };
    }
  }

  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  try {
    if (picker) {
      // ファイルへ直接書く（全体をメモリに載せない）
      const handle = await picker({ suggestedName: filename, types: [{ accept: { 'application/zip': ['.zip'] } }] });
      await downloadZip(files()).body!.pipeTo(await handle.createWritable());
    } else {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(await downloadZip(files()).blob());
      link.download = filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
    }
    return true;
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') return false;
    throw e;
  }
}
