"""Colab 用: Drive の写真から索引と配信用画像を作り、R2 にアップロードする。

使い方 (Colab, GPU ランタイム推奨):
    !pip install -q insightface onnxruntime-gpu pillow-heif boto3
    # このファイルの中身をセルに貼る（または %run build_index.py）
    prepare()          # Drive 認証・モデル読み込み・写真一覧
    start_build()      # バックグラウンドで 元画像DL → 縮小画像 → 顔の特徴量（途中から再開可）
    progress()         # 進捗
    write_index()      # /content/hff/index/meta.json, embeddings.bin を作る
    upload_r2()        # R2 へアップロード（Colab のシークレットに R2 の鍵を入れておく）

R2 のキー:
    orig/{id}.jpg       元画像（バイト列そのまま）
    large/{id}.webp     長辺 1600px
    thumb/{id}.webp     長辺 400px
    index/meta.json     { version, dim, photos: [{id,name,width,height,bytes}], facePhoto: [...] }
    index/embeddings.bin  int8 (顔の数 x 512) = round(L2正規化特徴量 * 127)
    index/embeddings_hq.bin  同じ顔の antelopev2 (glintr100) 特徴量。int8 (顔の数 x 512)

顔認識は buffalo_s (det_500m + w600k_mbf)。ブラウザ側 (web/src/face.ts) と同じモデル・前処理
（ブラウザと Python で同じ画像の特徴量が一致することを確認済み）。
索引側だけは集合写真の小さい顔・横顔を拾うため、元解像度の画像を antelopev2 の大きい検出器
(scrfd_10g, det_size 1280) で検出し、そのランドマークで切り出した顔を buffalo_s の認識モデルにかける。
さらに同じ顔 (同じ 5 点ランドマーク) を高精度モデル antelopev2 でも特徴量にしておき、
Worker は「buffalo_s で見つけた確実な顔 → その顔の高精度特徴量で探し直す」の 2 段で検索する。
しきい値は evaluate() で決めて meta.json に入れる。
"""

import hashlib
import io
import json
import os
import pickle
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

# 写真が入った Drive フォルダの ID。公開リポジトリには書かない（リンク共有のフォルダだと誰でも見られてしまう）
FOLDER_ID = os.environ.get('HFF_DRIVE_FOLDER_ID', '')  # Colab では prepare() の前に FOLDER_ID = '...' を代入する
OUT = '/content/hff'
# 小さすぎる顔・自信の低い検出は索引に入れない（顔の一辺は元画像のピクセル数）
MIN_FACE, MIN_DET = 48, 0.5
# 索引側は集合写真の小さい顔も拾うため、元解像度の画像を 1280 で検出し、切り出しも元解像度から行う。
# （ブラウザ側は自撮り 1 人なので 640 のままで良い。認識モデルが同じなので特徴量は比較できる）
INDEX_DET_SIZE = 1280
WORKERS = 6  # Drive の同時ダウンロード数。多すぎると 403 になる
LARGE, THUMB = 1600, 400
API = 'https://www.googleapis.com/drive/v3/files'

G = {}  # sess, app, images, state, progress


def photo_id(drive_id):
    """Drive の ID をそのまま外に出さないよう、ハッシュした 16 桁を使う"""
    return hashlib.sha256(f'hff:{drive_id}'.encode()).hexdigest()[:16]


def prepare():
    if not FOLDER_ID:
        raise SystemExit('FOLDER_ID (Drive フォルダの ID) を設定してください')
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from google.colab import auth
    from insightface.app import FaceAnalysis
    from pillow_heif import register_heif_opener
    import onnxruntime as ort

    register_heif_opener()
    auth.authenticate_user()
    sess = AuthorizedSession(google.auth.default()[0])
    sess.mount('https://', __import__('requests').adapters.HTTPAdapter(pool_maxsize=32))

    gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
    providers = (['CUDAExecutionProvider'] if gpu else []) + ['CPUExecutionProvider']
    import glob, shutil, urllib.request, zipfile
    from insightface.model_zoo import get_model

    # buffalo_s の認識モデル（ブラウザと同じ）
    FaceAnalysis(name='buffalo_s', allowed_modules=['detection'], providers=providers)  # 無ければダウンロード
    rec = get_model(os.path.expanduser('~/.insightface/models/buffalo_s/w600k_mbf.onnx'), providers=providers)
    rec.prepare(ctx_id=0 if gpu else -1)
    G['rec'] = rec
    print(f'buffalo_s rec ready ({"GPU" if gpu else "CPU"}), input_mean={rec.input_mean} std={rec.input_std}')

    # 高精度モデル antelopev2: 検出 (scrfd_10g) と認識 (glintr100)
    # 索引側の検出はブラウザと揃える必要がないので、横顔・小さい顔に強い大きい検出器を使う
    d = os.path.expanduser('~/.insightface/models/antelopev2')
    if not glob.glob(f'{d}/glintr100.onnx'):
        os.makedirs(d, exist_ok=True)
        urllib.request.urlretrieve(
            'https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip', f'{d}.zip')
        zipfile.ZipFile(f'{d}.zip').extractall(d)
        for p in glob.glob(f'{d}/**/*.onnx', recursive=True):
            if os.path.dirname(p) != d:
                shutil.move(p, d)
    hq = get_model(f'{d}/glintr100.onnx', providers=providers)
    hq.prepare(ctx_id=0 if gpu else -1)
    G['hq'] = hq
    app = FaceAnalysis(name='antelopev2', allowed_modules=['detection'], providers=providers)
    app.prepare(ctx_id=0 if gpu else -1, det_size=(INDEX_DET_SIZE, INDEX_DET_SIZE))
    print('antelopev2 (scrfd_10g + glintr100) ready')

    images, stack = [], [(FOLDER_ID, '')]
    while stack:
        fid, path = stack.pop()
        token = None
        while True:
            r = sess.get(API, params={
                'q': f"'{fid}' in parents and trashed = false",
                'fields': 'nextPageToken, files(id, name, mimeType, size)', 'pageSize': 1000,
                'pageToken': token, 'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'})
            r.raise_for_status()
            j = r.json()
            for f in j.get('files', []):
                if f['mimeType'] == 'application/vnd.google-apps.folder':
                    stack.append((f['id'], f"{path}{f['name']}/"))
                elif f['mimeType'].startswith('image/') or f['name'].lower().endswith(('.heic', '.heif')):
                    images.append({**f, 'path': path + f['name'], 'pid': photo_id(f['id'])})
            token = j.get('nextPageToken')
            if not token:
                break
    images.sort(key=lambda f: f['path'])
    for d in ['orig', 'large', 'thumb', 'index']:
        os.makedirs(f'{OUT}/{d}', exist_ok=True)
    G.update(sess=sess, app=app, images=images)
    print(f'{len(images)} 枚, {sum(int(f.get("size", 0)) for f in images) / 1e9:.1f} GB')


def _ckpt():
    return f'{OUT}/state.pkl'


def _fetch(f):
    """元画像を取得し、縮小画像を書き出す（済んでいればディスクから読む）"""
    pid = f['pid']
    orig = f'{OUT}/orig/{pid}.jpg'
    try:
        if not os.path.exists(orig):
            r = G['sess'].get(f"{API}/{f['id']}", params={'alt': 'media', 'supportsAllDrives': 'true'}, timeout=300)
            r.raise_for_status()
            with open(orig + '.part', 'wb') as fp:
                fp.write(r.content)
            os.replace(orig + '.part', orig)
        im = ImageOps.exif_transpose(Image.open(orig)).convert('RGB')
        w, h = im.size
        large = im.copy()
        large.thumbnail((LARGE, LARGE), Image.LANCZOS)
        large.save(f'{OUT}/large/{pid}.webp', 'WEBP', quality=85)
        thumb = large.copy()
        thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
        thumb.save(f'{OUT}/thumb/{pid}.webp', 'WEBP', quality=80)
        return f, im, (w, h), None
    except Exception as e:  # noqa: BLE001
        return f, None, None, repr(e)


def _worker():
    st, pr = G['state'], G['progress']
    try:
        todo = [f for f in G['images'] if f['pid'] not in st['photos']]
        with ThreadPoolExecutor(WORKERS) as ex:
            for k, (f, im, size, err) in enumerate(ex.map(_fetch, todo)):
                if not pr['running']:
                    break
                if err:
                    st['errors'][f['pid']] = err
                    continue
                faces = []
                # 元解像度で検出・切り出し (BGR)
                bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
                for fc in G['app'].get(bgr):  # 検出のみ (5 点ランドマーク付き)
                    x1, y1, x2, y2 = fc.bbox
                    e = G['rec'].get(bgr, fc)  # 同じランドマークで切り出して 2 つのモデルにかける
                    h = G['hq'].get(bgr, fc)
                    faces.append({'emb': (e / np.linalg.norm(e)).astype(np.float32),
                                  'hq': (h / np.linalg.norm(h)).astype(np.float32),
                                  'det': float(fc.det_score), 'side': float(min(x2 - x1, y2 - y1))})
                st['photos'][f['pid']] = {
                    'id': f['pid'], 'name': f['name'], 'drive_id': f['id'], 'width': size[0], 'height': size[1],
                    'bytes': os.path.getsize(f"{OUT}/orig/{f['pid']}.jpg"), 'faces': faces}
                st['errors'].pop(f['pid'], None)
                if k % 200 == 199:
                    pickle.dump(st, open(_ckpt(), 'wb'))
        pickle.dump(st, open(_ckpt(), 'wb'))
    except Exception:  # noqa: BLE001
        pr['err'] = traceback.format_exc()
    pr['running'] = False


def start_build(fresh=False):
    """fresh=True で顔の検出からやり直す（ダウンロード済みの元画像は使い回す）"""
    if fresh and os.path.exists(_ckpt()):
        os.remove(_ckpt())
    st = pickle.load(open(_ckpt(), 'rb')) if os.path.exists(_ckpt()) else {'photos': {}, 'errors': {}}
    G['state'] = st
    G['progress'] = {'start': time.time(), 'n0': len(st['photos']), 'running': True, 'err': None}
    G['thread'] = threading.Thread(target=_worker, daemon=True)
    G['thread'].start()
    print(f"開始: 残り {len(G['images']) - len(st['photos'])} 枚")


def progress(wait=60):
    G['thread'].join(timeout=wait)
    st, pr = G['state'], G['progress']
    n = len(st['photos']) - pr['n0']
    el = time.time() - pr['start']
    left = len(G['images']) - len(st['photos'])
    print(f"{len(st['photos'])}/{len(G['images'])} 枚, 顔 {sum(len(p['faces']) for p in st['photos'].values())}, "
          f"エラー {len(st['errors'])}, {el / max(n, 1):.2f} 秒/枚, 残り約 {left * el / max(n, 1) / 60:.0f} 分, "
          f"実行中={G['thread'].is_alive()}")
    if pr['err']:
        print(pr['err'])


THRESHOLDS = {'seed': 0.5, 'sure': 0.5, 'maybe': 0.4, 'hqSure': 0.45, 'hqMaybe': 0.35}  # evaluate() で更新


def _q8(x):
    """Worker と同じく int8 に量子化してから戻す"""
    return np.clip(np.round(x * 127), -127, 127) / 127


def _matrices():
    """索引に入る顔だけを並べた (buffalo_s 行列, 高精度行列, 写真 pid, 元の顔情報)"""
    st = G['state']
    E, H, pids, faces = [], [], [], []
    for f in G['images']:
        p = st['photos'].get(f['pid'])
        for fc in (p['faces'] if p else []):
            if fc['det'] >= MIN_DET and fc['side'] >= MIN_FACE:
                E.append(fc['emb'])
                H.append(fc['hq'])
                pids.append(f['pid'])
                faces.append(fc)
    return _q8(np.stack(E)).astype(np.float32), _q8(np.stack(H)).astype(np.float32), np.array(pids), faces


def write_index(thresholds=None):
    if thresholds:
        THRESHOLDS.update(thresholds)
    st = G['state']
    photos, face_photo, embs, hqs = [], [], [], []
    for f in G['images']:
        p = st['photos'].get(f['pid'])
        if not p:
            continue
        photos.append({k: p[k] for k in ['id', 'name', 'width', 'height', 'bytes']})
        for fc in p['faces']:
            if fc['det'] >= MIN_DET and fc['side'] >= MIN_FACE:
                face_photo.append(len(photos) - 1)
                embs.append(fc['emb'])
                hqs.append(fc['hq'])
    E = np.clip(np.round(np.stack(embs) * 127), -127, 127).astype(np.int8)
    H = np.clip(np.round(np.stack(hqs) * 127), -127, 127).astype(np.int8)
    E.tofile(f'{OUT}/index/embeddings.bin')
    H.tofile(f'{OUT}/index/embeddings_hq.bin')
    meta = {'version': time.strftime('%Y%m%d-%H%M%S'), 'dim': int(E.shape[1]), 'dimHq': int(H.shape[1]),
            'thresholds': THRESHOLDS, 'photos': photos, 'facePhoto': face_photo}
    json.dump(meta, open(f'{OUT}/index/meta.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print(f'索引: 写真 {len(photos)} 枚, 顔 {len(face_photo)} 個, しきい値 {THRESHOLDS}')


def search(query, t=None, use_hq=True, M=None, exclude_pid=None):
    """Worker (worker/src/search.ts) と同じ検索。{pid: (score, level)} を返す"""
    t = {**THRESHOLDS, **(t or {})}
    E, H, pids, _ = M or _matrices()
    q = query / np.linalg.norm(query)
    s1 = E @ q
    seeds = s1 >= t['seed']
    if exclude_pid is not None:  # 評価用: 問い合わせに使った写真自身はシードにしない
        seeds &= pids != exclude_pid
    if seeds.any() and use_hq:
        c = H[seeds].sum(0)
        s = H @ (c / np.linalg.norm(c))
        sure, maybe = t['hqSure'], t['hqMaybe']
    else:
        c = q + E[seeds].sum(0)
        s = np.maximum(s1, E @ (c / np.linalg.norm(c))) if seeds.any() else s1
        sure, maybe = t['sure'], t['maybe']
    best = {}
    for pid, v in zip(pids, s):
        if v >= maybe and v > best.get(pid, (-1,))[0]:
            best[pid] = (float(v), 'sure' if v >= sure else 'maybe')
    best.pop(exclude_pid, None)
    return dict(sorted(best.items(), key=lambda kv: -kv[1][0]))


def pseudo_identities(link=0.55, min_photos=6):
    """高精度特徴量で「確実に同一人物」の顔をつなぎ、評価用の正解グループを作る。
    つながりは厳しめ (link) にし、くっきり写った顔だけを使う。"""
    E, H, pids, faces = _matrices()
    clear = np.array([fc['det'] >= 0.7 and fc['side'] >= 80 for fc in faces])
    idx = np.where(clear)[0]
    S = H[idx] @ H[idx].T
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.where(np.triu(S >= link, 1))):
        parent[find(a)] = find(b)
    groups = {}
    for k, i in enumerate(idx):
        groups.setdefault(find(k), []).append(i)
    out = [g for g in groups.values() if len(set(pids[g])) >= min_photos]
    print(f'評価用グループ: {len(out)} 人 (写真 {min_photos} 枚以上), 顔 {sum(map(len, out))} 個')
    return out


def evaluate(groups, grid, queries_per_person=3, seed=0):
    """各グループから顔を 1 つ選んで「参加者の自撮り」の代わりにし、残りの写真をどれだけ見つけるか。
    正解 = そのグループの写真 + 高精度モデルでグループ平均に近い (>=0.5) 顔がある写真。
    grid: [(名前, しきい値 dict, use_hq)]。recall / 余計に出た写真の割合 を表示する。"""
    rng = np.random.default_rng(seed)
    M = _matrices()
    E, H, pids, faces = M
    rows = []
    for name, t, use_hq in grid:
        tp = fn = fp = tp_sure = fp_sure = 0
        for g in groups:
            c = H[g].sum(0)
            truth_faces = np.where(H @ (c / np.linalg.norm(c)) >= 0.5)[0]
            truth = set(pids[truth_faces]) | set(pids[g])
            for qi in rng.choice(g, size=min(queries_per_person, len(g)), replace=False):
                got = search(E[qi], t, use_hq, M, exclude_pid=pids[qi])
                want = truth - {pids[qi]}
                tp += len(want & set(got))
                fn += len(want - set(got))
                fp += len(set(got) - want)
                sure = {k for k, v in got.items() if v[1] == 'sure'}
                tp_sure += len(want & sure)
                fp_sure += len(sure - want)
        rows.append((name, tp / (tp + fn), fp / max(tp + fp, 1), tp_sure / (tp + fn), fp_sure / max(tp_sure + fp_sure, 1)))
        print(f'{name:28s} 見つけた割合 {rows[-1][1]:.3f}  余計 {rows[-1][2]:.3f}  | ほぼ確実のみ: 見つけた {rows[-1][3]:.3f} 余計 {rows[-1][4]:.3f}')
    return rows


def upload_r2(bucket='hackathon-photos', workers=32):
    """Colab のシークレット (鍵マーク) に R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY を入れておく"""
    import boto3
    from botocore.config import Config
    from google.colab import userdata

    s3 = boto3.client(
        's3', endpoint_url=f"https://{userdata.get('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=userdata.get('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=userdata.get('R2_SECRET_ACCESS_KEY'),
        region_name='auto', config=Config(max_pool_connections=workers, retries={'max_attempts': 5}))

    existing = set()
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket):
        existing.update(o['Key'] for o in page.get('Contents', []))

    jobs = []
    for p in G['state']['photos'].values():
        pid = p['id']
        jobs += [(f'orig/{pid}.jpg', f'{OUT}/orig/{pid}.jpg', 'image/jpeg'),
                 (f'large/{pid}.webp', f'{OUT}/large/{pid}.webp', 'image/webp'),
                 (f'thumb/{pid}.webp', f'{OUT}/thumb/{pid}.webp', 'image/webp')]
    jobs = [j for j in jobs if j[0] not in existing]
    print(f'アップロード {len(jobs)} 個（済み {len(existing)} 個）')

    done = [0]

    def put(job):
        key, path, ctype = job
        s3.upload_file(path, bucket, key, ExtraArgs={'ContentType': ctype})
        done[0] += 1
        if done[0] % 500 == 0:
            print(f'  {done[0]}/{len(jobs)}')

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(put, jobs))
    # 索引は最後に置く（画像が揃ってから検索に出るように）
    for name in ['embeddings.bin', 'embeddings_hq.bin']:
        s3.upload_file(f'{OUT}/index/{name}', bucket, f'index/{name}',
                       ExtraArgs={'ContentType': 'application/octet-stream'})
    s3.upload_file(f'{OUT}/index/meta.json', bucket, 'index/meta.json',
                   ExtraArgs={'ContentType': 'application/json'})
    print('R2 アップロード完了')
