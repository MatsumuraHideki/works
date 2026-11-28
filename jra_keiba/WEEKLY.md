# 毎週の手順（重賞の直前予想と答え合わせ）

netkeiba の取得はすべて GitHub Actions 経由です（トリガーファイルを push すると動きます）。
取得間隔は縮めないこと。HTTP 400 が出たら取得を止めて報告します。

## 金曜の夜：準備

1. `git pull --rebase`。`python3 -c "import lightgbm"` が失敗したら `pip install -q -r jra_keiba/requirements.txt`。
2. **先週末の結果が入っているか確認**（`data/raw/results/` の今月分）。入っていなければ、
   `jra-data` が実行中でないことを確かめてから `scrape_plan.json` の `chunks` を次のようにして push、
   bot のコミットが止まるのを待って pull。

   ```json
   [{"cmd": "results", "from": "<今月 YYYY-MM>", "to": "<先月 YYYY-MM>", "max_minutes": 60},
    {"cmd": "odds", "from": "<今月>", "to": "<先月>", "max_minutes": 60, "args": "--op-class --types 1,4,5,7"},
    {"cmd": "horses", "from": "<今月>", "to": "2010-01", "max_minutes": 180}]
   ```

   3 つ目は JRA 以外（地方・海外）の出走の読み直しで、前回読んだあとに JRA で走った馬だけが対象
   （週に 1,000〜1,500 頭、1〜1.5 時間）。週明けの月曜〜木曜に回すのがよい。
3. **血統のカバー率を確認**。

   ```bash
   python3 -W ignore -c "from keiba import features as f; r=f.load_raw()[1]; print(f.pedigree_usable(r, f.load_pedigree()))"
   ```

   - `True` になっていたら全体を再学習（`pipeline.fit_all()` → `save_artifacts` → `write_report` /
     `strategy.write_strategy_report` / `summary.write_all`、約 50 分）。テストを通し、
     `cp data/work/model.pkl models/model.pkl` して reports とともにコミット。
     回収率が急に跳ね上がった場合は情報の漏れを疑い、採用しない。
   - `False` のままで `jra-data` が動いていなければ、`scrape_plan.json` に
     `{"cmd": "pedigree", "from": "<今月>", "to": "2010-01", "max_minutes": 330}` を数個入れて月曜以降に回す。
     金〜日は週末の取得と重なるので血統収集を始めない。
4. 未精算の予想があれば `python3 -W ignore -m keiba.settle`。
5. **今週末の出馬表**: `predict_trigger.json` を `dates: ""`（今後 6 日の開催日すべて）、`grades: "G1,G2,G3"`、
   `cards_only: true`、`results: ""`、`requested_at: <今>` にして push →「Update race cards and odds」の
   コミットを待って pull → `data/cards/*.json` から今週末の重賞（race_id・レース名・日付・発走時刻）を列挙。
   日曜分がまだ無ければ土曜の夜に再確認。
6. 重賞ごとに予約を 2 つ入れる: **発走 17 分前（直前予想）** と **発走 25 分後（答え合わせ）**。
   あわせて **土曜・日曜の 9:15** に「オッズの記録開始」の予約を入れる（下の節）。
7. 対象レースと予想を出す時刻を報告。

## 土曜・日曜の 9:15：オッズの記録開始

`watch_trigger.json` を `{"date": "", "lead": "30,15,10,5", "requested_at": "<今>"}` にして push する
（`date` が空ならその日）。`jra-odds-watch` がその日の全レースについて発走 30・15・10・5 分前の単勝オッズを
`data/raw/odds_snapshots/` に記録する（1 レース 1 リクエスト）。確定オッズは翌週の結果取り込みで入るので、
データがたまったら「予想時→確定」のずれをオッズ帯ごとに集計し、`reports/experiments.md` の
「1 割ずれ・2 割ずれ」のどちらが実態に近いかを確かめる。

## 発走 17 分前：直前予想

1. `git pull --rebase`。
2. `predict_trigger.json` を `dates: "<YYYYMMDD>"`、`cards_only: true`、`results: ""`、`requested_at: <今>` にして push。
3. 「Update race cards and odds」のコミットを待って pull（約 1 分半）。
4. `python3 -W ignore -m keiba.predict --budget 5000 --race-ids <race_id>`（モデルは `models/model.pkl`）。
5. `reports/predictions/<日付>_<race_id>.md` を読んですぐ報告:
   オッズの時刻、買い目 A（単勝・期待値 1.10 以上。無ければ「見送り」と明言）、モデル 1 位の馬と期待値、
   単勝の期待値 0.95 以上の馬、買い目 B（5,000 円、まとめ買いの形で、過去検証の回収率も添える）。
6. reports をコミット・プッシュ。

## 発走 25 分後：答え合わせ

1. `git pull --rebase`。
2. `predict_trigger.json` の `results` を `"<race_id>"` にして push →「Update race cards and odds」を待って pull。
   `data/cards/results/<race_id>.json` が無ければ数分後にもう一度。
3. `python3 -W ignore -m keiba.settle`。
4. 報告: 1〜3 着、買い目 A・B の当たり外れと払戻、通算成績、モデル上位馬の着順、
   予想時から確定までのオッズの動き。外れても事実をそのまま伝える。
5. reports をコミット・プッシュし、`results` を `""` に戻してコミット・プッシュ。
