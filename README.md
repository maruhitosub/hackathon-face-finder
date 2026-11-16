# Hackathon Face Finder

ハッカソンの写真の中から、顔写真 1 枚で自分が写っている写真を探して一覧・ダウンロードできるサイト。
Cloudflare Workers + R2。共通パスワードで保護。しくみの解説は [GUIDE.md](GUIDE.md)。

```
web/src/
  main.ts          画面の切り替え（ログイン → 顔写真 → 結果）
  views/           各画面: login / upload / results / viewer（拡大表示）
  face.ts          顔検出・位置合わせ・特徴量（onnxruntime-web, buffalo_s）
  api.ts           Worker の API クライアント（api.mock.ts はバックエンド無しで動かす偽物）
  zip.ts           元画像をまとめて ZIP 保存
worker/src/
  index.ts         ルーティングと login / search
  auth.ts          パスワード照合・署名付き Cookie
  search.ts        索引の読み込みと 2 段階の照合
  photos.ts        R2 の写真を返す
preprocess/        Colab で索引と配信用画像を作り R2 に上げるスクリプト
scripts/           モデル取得・ローカル用の偽データ
```

```
ブラウザ ── 自撮りを 512 個の数値に変換（写真は送らない）
   │  POST /api/search
Worker ── R2 の索引と照合 → 写真の一覧 / 画像を返す（Cookie で認証）
   │
R2 ── orig/ large/ thumb/ index/   ← Colab (preprocess/) が事前に作る
```

## ローカルで動かす

```bash
npm install && npm --prefix web install
npm run models        # 顔認識モデルを web/public/models に置く
npm run seed:local    # 偽の写真 80 枚と索引をローカル R2 に入れる (.dev.vars も作る。パスワード: hackathon)
npm run dev           # フロントをビルドして wrangler dev → http://localhost:8787
```

- 画面だけ触るなら `npm --prefix web run dev:mock`（API もブラウザ内の偽物。http://localhost:5173）
- フロントを編集しながら本物の API で試すなら、`npx wrangler dev` と `npm --prefix web run dev` を両方起動（Vite が /api などを 8787 に流す）
- 偽データの顔は本物の顔ではないので、ローカルで自分の顔を貼っても「見つかりませんでした」になる。検索 API の確認は
  `curl -b cookie -X POST -H 'content-type: application/json' --data @.seed/query-a.json localhost:8787/api/search`

## 本番データを作る (Colab)

1. Colab を GPU ランタイムにして `preprocess/build_index.py` の中身をセルに貼る
2. `!pip install -q insightface onnxruntime-gpu pillow-heif boto3`
3. `FOLDER_ID = "写真フォルダの Drive ID"` を代入 → `prepare()` → `start_build()` → `progress()`（2,814 枚・16GB。元画像を Drive から落とし、縮小画像と顔の特徴量を作る。L4 で約 9 分）
4. 精度の確認: `groups = pseudo_identities()` → `evaluate(groups, grid)` でしきい値ごとの成績を見る
5. `write_index({...しきい値})`（しきい値は meta.json に入り、Worker はそれを使う）
6. R2 の API トークン（User API Token で可。対象バケットへの Object Read & Write）を作り、Colab のシークレットに
   `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` を登録 → `upload_r2()`
   （シークレットは Colab の画面からセルを実行したときしか読めない）

`search(embedding)` で Worker と同じ検索を Colab 上で試せる。

## 検索のしくみ

1. ブラウザで自撮りを buffalo_s (軽いモデル) で 512 次元に変換して送る
2. Worker が全顔の buffalo_s 特徴量と比べ、0.5 以上を「シード」（確実に本人）にする
3. シードの顔の **antelopev2 (高精度モデル) の特徴量**の平均で全顔を探し直す。0.5 以上=ほぼ確実、0.4 以上=たぶん
4. シードが無いときは buffalo_s だけで判定（0.5 / 0.4）

高精度モデル (約 250MB) はブラウザでは動かせないので、索引側で事前計算した特徴量だけを照合に使う。
索引側の顔検出も antelopev2 の検出器 (scrfd_10g, det 1280, 元解像度) で、横顔・小さい顔まで拾う。

## 検証結果 (2026-11-16, Colab L4)

- ブラウザ (`face.ts`) と Python (insightface) の特徴量: 同じ画像で類似度 1.0000（モデルファイルのハッシュも一致）
- 索引: 写真 2,814 枚・検出した顔 11,622 個（うち索引に入れた顔 10,218 個）
- 126 人分の模擬検索（高精度モデルで確実に同一人物とつながる顔のグループを正解とし、その中の顔を自撮りの代わりにする）:
  - 軽いモデルのみ: 見つけた割合 89.9%・別人の混入 11.5%
  - **2 段階 (採用)**: 見つけた割合 94.3%・別人の混入 7.2%。「ほぼ確実」だけなら 90.3%・混入 0.4%
- 50500.jpg（全体写真をサイトの画面から貼った場合、本物の索引）: 72 枚ヒット（ほぼ確実 70・たぶん 2）。
  前回 antelopev2 だけで選んだ 72 枚のうち 70 枚と一致。新規 2 枚はいずれも同じ連写の中の「たぶん」
- 模擬検索の正解は高精度モデル由来なので、公開前に何人か本人の自撮りで試してしきい値を見直す

## 注意

- insightface の学習済みモデル (buffalo_s) は**非商用・研究目的**のライセンス
- 顔の特徴量は個人情報。公開期間が終わったら R2 のバケットごと削除する。削除依頼があればその人の顔を索引から外して `write_index()` → `index/` を上げ直す
- 参加者が貼った顔写真はブラウザから出ない。サーバーに届くのは 512 個の数値だけで、保存もしない
