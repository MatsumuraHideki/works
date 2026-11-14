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
