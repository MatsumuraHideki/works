"""Race-day paths: card parsing details, protection of published predictions, settlement."""
import datetime as dt
import json

from keiba import netkeiba as nk
from keiba import predict, settle

JST = dt.timezone(dt.timedelta(hours=9))


def test_card_going_forms():
    assert nk.card_going("15:30発走 / ダ1600m 左\n\n/ 予報 良 / 直線:弱横風") == "良"
    assert nk.card_going("15:40発走 / 芝2200m 右 外 B\n\n/ 予報 / 直線:弱向風") is None      # not announced yet
    assert nk.card_going("15:30発走 / 芝2500m 左 B\n\n/ 稍 / 直線:弱追風") == "稍重"
    assert nk.card_going("10:00発走 / ダ1200m 右 / 天候:雨 / 馬場:不良") == "不良"
    assert nk.card_going("10:00発走 / 芝1000m 直線 / 予報 重 / 直線:無風") == "重"       # 直線 course, wind ignored


def _card(rid, date, start, fetched):
    return {"race": {"race_id": rid, "name": "t", "date": date, "start_time": start}, "runners": [],
            "fetched_at": fetched, "odds": {}, "odds_status": "middle"}


def test_races_past_post_time_are_not_predicted_again(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "CARDS", tmp_path)
    (tmp_path / "a.json").write_text(json.dumps(_card("A", "2026-11-14", "15:30", "2026-11-14T15:13:57+09:00")))
    (tmp_path / "b.json").write_text(json.dumps(_card("B", "2026-11-15", "15:40", "2026-11-14T15:32:13+09:00")))
    # snapshot taken after the race started: never a prediction
    (tmp_path / "c.json").write_text(json.dumps(_card("C", "2026-11-15", "10:00", "2026-11-15T10:02:00+09:00")))
    before = dt.datetime(2026, 11, 14, 15, 20, tzinfo=JST)
    after = dt.datetime(2026, 11, 14, 15, 31, tzinfo=JST)
    assert [c["race"]["race_id"] for c in predict.load_cards(now=before)] == ["A", "B"]
    assert [c["race"]["race_id"] for c in predict.load_cards(now=after)] == ["B"]
    assert [c["race"]["race_id"] for c in predict.load_cards(["A"], now=after)] == []      # even when asked by id
    assert predict.is_pre_race(_card("A", "2026-11-14", "15:30", "2026-11-14T15:29:00+09:00"))
    assert not predict.is_pre_race(_card("A", "2026-11-14", "15:30", "2026-11-14T15:30:00+09:00"))


def test_settlement_scores_only_pre_race_predictions(tmp_path, monkeypatch):
    pred, live = tmp_path / "pred", tmp_path / "live"
    pred.mkdir(); live.mkdir()
    monkeypatch.setattr(settle, "PRED", pred)
    monkeypatch.setattr(settle, "LIVE", live)
    monkeypatch.setattr(settle.features, "load_raw", lambda: (None, None, None))
    race = {"race_id": "R1", "name": "テスト", "grade": "G3", "date": "2026-11-14", "start_time": "15:30"}
    slip = [{"type": "単勝", "combo": [5], "stake": 300, "odds": 4.0},
            {"type": "ワイド", "combo": [7, 5], "stake": 100, "odds": 12.0},
            {"type": "三連複", "combo": [1, 2, 3], "stake": 100, "odds": 50.0}]
    base = {"race": race, "runners": [{"number": 5, "horse": "x"}], "slip": slip, "win_slip": slip[:1]}
    (pred / "20261114_R1.json").write_text(json.dumps({**base, "fetched_at": "2026-11-14T15:14:00+09:00"}))
    (live / "R1.json").write_text(json.dumps({"race_id": "R1", "order": [{"finish": "1", "number": 5, "horse_name": "x"}],
        "payouts": [{"bet_type": "単勝", "combo": [5], "payout": 420}, {"bet_type": "ワイド", "combo": [5, 7], "payout": 1310}]}))
    rows = settle.settle_all(log=lambda *a: None)
    assert len(rows) == 1 and rows[0]["stake"] == 500
    assert rows[0]["return"] == 300 * 4.2 + 100 * 13.1           # payouts are per 100 yen; ワイド key is unordered
    assert rows[0]["a_stake"] == 300 and rows[0]["a_return"] == 1260
    # the same file regenerated after the race must not be scored
    (pred / "20261114_R1.json").write_text(json.dumps({**base, "fetched_at": "2026-11-14T15:31:00+09:00"}))
    assert settle.settle_all(log=lambda *a: None) == []


def test_day_bias_of_result_uses_draw_and_last3f_rank():
    """Race-day track bias: mean draw and last-3F rank of the first three, scratches left out."""
    from keiba import netkeiba as nk
    order = [{"finish": "1", "number": 1.0, "last3f": 34.0}, {"finish": "2", "number": 5.0, "last3f": 35.0},
             {"finish": "3", "number": 3.0, "last3f": 36.0}, {"finish": "4", "number": 2.0, "last3f": 34.5},
             {"finish": "5", "number": 4.0, "last3f": 37.0}, {"finish": "取消", "number": 6.0, "last3f": None}]
    b = nk.day_bias_of_result({"order": order})
    assert abs(b["inner"] - (0 + 4 + 2) / 3 / 4) < 1e-9          # five starters: (number - 1) / 4
    assert abs(b["closer"] - (0 + 2 + 3) / 3 / 4) < 1e-9         # last-3F ranks 1, 3, 4 of 5
    assert nk.day_bias_of_result({"order": order[:3]}) is None   # too few starters to mean anything
    bare = [{**o, "last3f": None} for o in order]                 # times not posted yet
    b = nk.day_bias_of_result({"order": bare})
    assert b["closer"] is None and abs(b["inner"] - 0.5) < 1e-9


def test_horse_collection_survives_a_site_timeout(tmp_path, monkeypatch):
    """A request that times out must not end the collection or lose what was already read."""
    import argparse
    import pandas as pd
    import requests
    from keiba import netkeiba as nk
    from keiba import scrape

    raw = tmp_path / "raw"
    (raw / "results").mkdir(parents=True)
    (raw / "races").mkdir(parents=True)
    pd.DataFrame({"race_id": ["202605050101"] * 3, "horse_id": ["h1", "h2", "h3"]}).to_csv(
        raw / "results" / "202611.csv.gz", index=False)
    pd.DataFrame({"race_id": ["202605050101"], "date": ["2026-11-01"]}).to_csv(raw / "races" / "202611.csv.gz", index=False)
    monkeypatch.setattr(scrape, "RAW", raw)

    def fake_fetch(f, url, referer=None):
        if "h2" in url:
            raise requests.exceptions.ReadTimeout("read timed out")
        return url

    run = {"horse_id": None, "date": "2026-10-01", "venue": "大井", "jra": False, "race_name": "x", "n_runners": 10.0,
           "number": 1.0, "win_odds": 2.0, "popularity": 1.0, "finish": 1.0, "finish_raw": "1", "surface": "ダ",
           "distance": 1600, "going": "良", "time_sec": 100.0, "passing": "1-1", "last3f": 38.0}
    monkeypatch.setattr(scrape, "fetch_text", fake_fetch)
    monkeypatch.setattr(nk, "parse_horse_runs", lambda html, hid: [dict(run, horse_id=hid)])
    args = argparse.Namespace(interval=0.0, max_minutes=1.0, since=2000, push=False, push_every=40)
    assert scrape.cmd_horses(args) == 0
    state = scrape.load_state()
    assert set(state["horse_runs_done"]) == {"h1", "h3"}            # h2 is left for a later run
    saved = pd.read_csv(raw / "horse_runs.csv.gz", dtype={"horse_id": str})
    assert set(saved["horse_id"]) == {"h1", "h3"}
