"""Predict upcoming races from card snapshots and write betting reports.

Usage:
    python -m keiba.predict [--race-ids a,b] [--bankroll 100000] [--min-ev 1.15] [--retrain]

Reads data/cards/*.json (see keiba.cards) for races that have not been run yet,
writes reports/predictions/<date>_<race_id>.md and .json plus reports/latest.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import betting, exotic, features, model, pipeline, strategy
from .scrape import today_jst

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "cards"
OUT = ROOT / "reports" / "predictions"
ASPECTS = {"ability": "能力", "form": "近走", "aptitude": "適性", "people": "騎手厩舎",
           "pedigree": "血統", "race": "展開・枠"}


def load_cards(race_ids: list[str] | None = None, since: dt.date | None = None) -> list[dict]:
    cards = []
    for p in sorted(CARDS.glob("*.json")):
        c = json.loads(p.read_text())
        if race_ids and c["race"]["race_id"] not in race_ids:
            continue
        if not race_ids and since and c["race"]["date"] < since.isoformat():
            continue
        cards.append(c)
    return cards


def cards_to_frames(cards: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races = pd.DataFrame([c["race"] for c in cards])
    runners = pd.DataFrame([r for c in cards for r in c["runners"] if not r.get("cancelled")])
    for col in ("finish", "time_sec", "last3f", "win_odds", "popularity", "prize"):
        runners[col] = np.nan
    runners["finish_raw"] = ""
    runners["passing"] = None
    ped = pd.DataFrame([p for c in cards for p in c.get("pedigree", [])],
                       columns=["horse_id", "sire", "dam", "bms"])
    return races, runners, ped


def card_odds(card: dict, rows: pd.DataFrame) -> tuple[dict[str, dict], pd.Series, str]:
    """Odds tables keyed by 馬番 tuples, the win odds per runner, and their kind.

    kind: 'real' (sales open / final), 'yoso' (netkeiba's predicted odds, win only), 'none'.
    """
    status = card.get("odds_status")
    raw = card.get("odds", {})
    win = pd.Series(np.nan, index=rows.index)
    if not raw.get("単勝") or status in (None, "NG"):
        return {}, win, "none"
    w = {int(k): v[0] for k, v in raw["単勝"].items()}
    if status == "yoso":
        win = rows["row_id"].astype(float).map(w)
        return {}, win, "yoso"
    win = rows["number"].map(w)
    tables = {bt: {pipeline.parse_combo(bt, k): float(v[0]) for k, v in tab.items()} for bt, tab in raw.items()}
    return tables, win, "real"


def hours_to_post(card: dict) -> float | None:
    """Hours between the odds snapshot and the scheduled post time (both JST)."""
    r = card["race"]
    try:
        post = dt.datetime.fromisoformat(f"{r['date']}T{r['start_time']}:00")
        snap = dt.datetime.fromisoformat(str(card.get("odds_time") or card["fetched_at"])[:19].replace(" ", "T"))
    except (KeyError, TypeError, ValueError):
        return None
    return (post - snap).total_seconds() / 3600


def mark(values: pd.Series) -> pd.Series:
    """Within-race marks for an aspect score: ◎ best, ○ top quarter, △ above median."""
    if values.nunique() <= 1:          # no information for this aspect (e.g. pedigree not collected yet)
        return pd.Series("－", index=values.index)
    r = values.rank(ascending=False, method="min")
    n = len(values)
    return pd.Series(np.where(r == 1, "◎", np.where(r <= max(2, n * 0.25), "○",
                     np.where(r <= n * 0.5, "△", "－"))), index=values.index)


WIN_EV = 0.95      # 単勝 threshold whose backtest is reported in reports/summary.json
TOP_PICK_SHARE = 0.4   # share of a budget put on the model's top pick when its win EV >= 1.0


def predict_race(card: dict, rows: pd.DataFrame, art: dict, boosters: dict, bankroll: int, min_ev: float,
                 budget: int = 0) -> dict:
    feats = art["feats"]
    rows = rows.copy()
    tables, win_odds, kind = card_odds(card, rows)
    rows["odds"] = win_odds
    use_market = kind != "none" and rows["odds"].notna().all()
    rows = model.add_market_features(rows, "odds")       # NaN when there are no odds
    s = model.predict_stage1(boosters, rows, feats)
    rows["s_win"], rows["s_top3"], rows["s_res"] = s["s_win"], s["s_top3"], s["s_res"]
    rows["p_fund"] = model.Stage2(["s_win", "s_top3"], np.asarray(art["s2_fund"])).predict(rows)
    if use_market:
        rows["p"] = model.Stage2(pipeline.FULL_COLS, np.asarray(art["s2_full"])).predict(rows)
    else:
        rows["p"] = rows["p_fund"]
    rows["ev"] = rows["p"] * rows["odds"]
    contrib = model.group_contributions(boosters["win"], rows[feats], art["groups"])
    for g in ASPECTS:
        rows[f"a_{g}"] = mark(contrib[g]) if g in contrib else "－"

    drawn = rows["number"].notna().all()
    bets: list[betting.Bet] = []
    slip: list[betting.Bet] = []
    top_by_type: dict[str, list[betting.Bet]] = {}
    if kind == "real" and drawn and len(rows) >= 2:
        probs = betting.bet_probabilities(rows["p"].to_numpy(), [int(x) for x in rows["number"]],
                                          art["lam2"], art["lam3"])
        cand = betting.evaluate_bets(probs, tables)
        # Same rule as the backtest (reports/backtest.md section 4): every 馬連 / ワイド / 三連複
        # ticket whose model expected value reaches the threshold, 100 yen each. Buying only
        # the few highest-EV tickets tested much worse than buying the whole set.
        bets = sorted((b for b in cand if b.bet_type in exotic.CORE and b.ev >= min_ev and b.prob >= 0.005),
                      key=lambda b: (exotic.CORE.index(b.bet_type), -b.ev))
        for b in bets:
            b.stake = 100
        if budget:
            wins = [b for b in cand if b.bet_type == "単勝" and b.ev >= WIN_EV and b.prob >= 0.01]
            pool = [betting.Bet(b.bet_type, b.combo, b.prob, b.odds, b.ev, b.kelly) for b in wins + bets]
            # Best-supported rule in the backtest: the model's top pick when its own win EV is
            # 1.0 or more. When it applies it gets a fixed share of the budget up front.
            top = rows.loc[rows["p"].idxmax()]
            anchor = 0
            if pd.notna(top["ev"]) and top["ev"] >= 1.0:
                anchor = int(budget * TOP_PICK_SHARE // 100) * 100
            slip = betting.budget_slip(pool, budget - anchor)
            if anchor:
                key = (int(top["number"]),)
                hit = next((b for b in slip if b.bet_type == "単勝" and b.combo == key), None)
                if hit is None:
                    hit = betting.Bet("単勝", key, float(top["p"]), float(top["odds"]), float(top["ev"]), 0.0)
                    slip.insert(0, hit)
                hit.stake += anchor
                slip.sort(key=lambda b: -b.prob)
        for bt in betting.BET_TYPES:
            top = sorted((b for b in cand if b.bet_type == bt and b.prob >= 0.005),
                         key=lambda b: b.ev, reverse=True)[:5]
            if top:
                top_by_type[bt] = top
    # tickets of the strategies that held up best in the backtest, applied to this race
    g = rows.assign(p_final=rows["p"], win_odds=rows["odds"])
    inv = 1.0 / g["win_odds"]
    g["p_market"] = inv / inv.sum() if g["win_odds"].notna().all() else np.nan
    path = ROOT / "reports" / "strategy_top.json"
    stored = json.loads(path.read_text()) if path.exists() else []
    validated = strategy.apply_strategies(stored, g, card["race"]["race_id"]) if kind != "yoso" else []
    return {"card": card, "rows": rows.sort_values("p", ascending=False), "kind": kind, "validated": validated, "slip": slip, "budget": budget,
            "use_market": use_market, "drawn": drawn, "bets": bets, "top_by_type": top_by_type}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render(res: dict, art: dict, bankroll: int, min_ev: float) -> str:
    c, rows, kind = res["card"], res["rows"], res["kind"]
    r = c["race"]
    grade = f"（{r['grade']}）" if r.get("grade") else ""
    out = [f"## {r['name']}{grade} {r['date']} {r.get('place', '')}{r['race_no']}R "
           f"{r.get('surface', '')}{r.get('distance', '')}m {r.get('start_time') or ''}発走\n"]
    odds_note = {"real": f"発売中/確定オッズ（{c.get('odds_time') or c.get('fetched_at')} 時点）",
                 "yoso": "netkeiba の予想オッズ（発売前。実際のオッズではありません）",
                 "none": "オッズ未発表"}[kind]
    out.append(f"- オッズ: {odds_note}")
    hrs = hours_to_post(c)
    if kind == "real" and hrs is not None and hrs > 6:
        out.append(f"- **発走まで約 {hrs:.0f} 時間あり、まだ票数が少ない段階のオッズです。**"
                   "この時点のオッズは少額の投票で大きく動き、券種どうしの整合も取れていないため、"
                   "下の期待値と買い目は当日のオッズで出し直すまで目安にとどめてください。")
    if not res["drawn"]:
        out.append(f"- 枠順確定前（登録 {len(rows)} 頭）。出走馬と枠順が確定すると確率は変わります。")
    out.append(f"- データ取得: {c.get('fetched_at')} / モデル学習範囲: 〜{art['trained_through']}\n")

    head = ["馬番", "馬名", "騎手", "勝率", "モデル単独", "単勝オッズ", "期待値", "買える下限オッズ"] + list(ASPECTS.values())
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for _, x in rows.iterrows():
        num = str(int(x["number"])) if pd.notna(x["number"]) else "-"
        odds = f"{x['odds']:.1f}" if pd.notna(x["odds"]) else "-"
        ev = f"{x['ev']:.2f}" if pd.notna(x["ev"]) else "-"
        need = f"{min_ev / x['p_fund' if not res['use_market'] else 'p']:.1f}"
        cells = [num, str(x["horse_name"]), str(x.get("jockey_name") or ""), _pct(x["p"]), _pct(x["p_fund"]),
                 odds, ev, need] + [x[f"a_{g}"] for g in ASPECTS]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append(f"「勝率」は{'モデルとオッズを統合した' if res['use_market'] else 'モデル単独の'}推定値。"
               f"「買える下限オッズ」は単勝の期待値が {min_ev:.2f} になるオッズで、これ以上つけば買い。"
               "観点別の印は、その観点がこの馬の評価をどれだけ押し上げているかのレース内順位（◎最上位 ○上位 △中位 －下位）。\n")

    if kind == "real":
        # 単勝: the threshold whose backtest is in reports/summary.json
        thr = WIN_EV
        picks = rows[(rows["ev"] >= thr) & (rows["p"] >= 0.01)]
        out.append(f"### 単勝で期待値 {thr:.2f} 以上の馬\n")
        sp = ROOT / "reports" / "summary.json"
        hist = next((r for r in json.loads(sp.read_text()) if r["label"] == f"単勝 期待値{thr:.2f}以上"), None) \
            if sp.exists() else None
        if hist:
            out.append(f"過去検証（同じ基準、{hist['races']:,} レース・{hist['tickets']:,} 点）: 的中率 {_pct(hist['hit_rate'])}、"
                       f"回収率 {_pct(hist['roi'])}（偶然の幅 {_pct(hist['ci_low'])}〜{_pct(hist['ci_high'])}）。"
                       "確定オッズでの検証なので、買った後にオッズが下がると期待値は下がります。\n")
        if len(picks):
            for _, x in picks.iterrows():
                num = str(int(x["number"])) if pd.notna(x["number"]) else "-"
                out.append(f"- {num} {x['horse_name']}（勝率 {_pct(x['p'])}、{x['odds']:.1f} 倍、期待値 {x['ev']:.2f}）")
            out.append("")
        else:
            out.append("該当なし。\n")
    if kind == "real" and res["drawn"] and res.get("budget"):
        out.append(f"### 予算 {res['budget']:,} 円の買い目\n")
        if res["slip"]:
            out.append("単勝は期待値 0.95 以上、馬連・ワイド・三連複は期待値 "
                       f"{min_ev:.2f} 以上の買い目を、当たりやすい順に 100 円ずつ入れ、余った分を上から足しています。"
                       "モデルの 1 位の馬が単勝の期待値 1.0 以上のときは、予算の 4 割を先にその単勝に充てます"
                       "（検証で最も成績が良かった買い方のため）。\n")
            out.append("| 券種 | 買い目 | 購入額 | オッズ | 的中確率 | 当たった場合の払戻 |")
            out.append("|---|---|---|---|---|---|")
            for b in res["slip"]:
                out.append(f"| {b.bet_type} | {b.label()} | {b.stake:,}円 | {b.odds:.1f} | {_pct(b.prob)} | "
                           f"{b.stake * b.odds:,.0f}円 |")
            total = sum(b.stake for b in res["slip"])
            out.append(f"\n合計 {len(res['slip'])} 点 {total:,} 円。"
                       "過去の検証では、この選び方の回収率は単勝・馬連・ワイド・三連複とも 93〜96% 前後で、"
                       "選ばずに買う場合（72〜80%）よりは良いものの、プラスになるとは言えていません。\n")
        else:
            out.append("基準を満たす買い目がありません。このレースは見送りです。\n")
    if kind == "real" and res["drawn"]:
        out.append(f"### 期待値で選んだ買い目（馬連・ワイド・三連複、期待値 {min_ev:.2f} 以上を各 100 円）\n")
        evp = ROOT / "reports" / "ev_backtest.json"
        ev = json.loads(evp.read_text()) if evp.exists() else None
        if ev:
            base = f"。選ばずに買った場合は {_pct(ev['baseline_roi'])}" if ev.get("baseline_roi") else ""
            out.append(f"過去検証（同じルール、{ev['races']} レース・{ev['tickets']:,} 点）: 的中率 {_pct(ev['hit'])}、"
                       f"回収率 {_pct(ev['roi'])}（偶然の幅 {_pct(ev['ci_low'])}〜{_pct(ev['ci_high'])}）{base}。"
                       "偶然の幅の下限が 100% を超えるまでは、プラスと言い切れない買い方です。"
                       "期待値が特に高い数点だけに絞ると成績が悪化したため、該当する買い目を全部買う前提です。\n")
        else:
            out.append("**この選び方はまだ過去データで検証できていません。**参考として見てください。\n")
        if res["bets"]:
            for bt in exotic.CORE:
                sel = [b for b in res["bets"] if b.bet_type == bt]
                if sel:
                    out.append(f"- **{bt}**（{len(sel)} 点）: " + "、".join(f"{b.label()}({b.odds:.0f}倍)" for b in sel[:40])
                               + (f" ほか{len(sel) - 40}点" if len(sel) > 40 else ""))
            stake, ret = betting.portfolio_ev(res["bets"])
            out.append(f"\n合計 {len(res['bets'])} 点 {stake:,} 円。\n")
        else:
            out.append("基準を満たす買い目なし。\n")
        out.append("### 券種別の期待値上位\n")
        out.append("| 券種 | 買い目 | 的中確率 | オッズ | 期待値 |")
        out.append("|---|---|---|---|---|")
        for bt, top in res["top_by_type"].items():
            for b in top[:3]:
                out.append(f"| {bt} | {b.label()} | {_pct(b.prob)} | {b.odds:.1f} | {b.ev:.2f} |")
        out.append("")
    if res.get("validated"):
        out.append("### 過去の検証で成績が良かった買い方を当てはめた買い目\n")
        out.append("| 区分 | 買い方 | 選び方 | 条件 | 買い目 | 検証回収率 | 前半/後半 | 偶然の幅 | レース数 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for v in res["validated"]:
            sep = "→" if v["bet_type"] in ("馬単", "三連単") else "-"
            tix = "、".join(sep.join(map(str, t)) for t in v["tickets"][:10])
            tix += f" ほか{len(v['tickets']) - 10}点" if len(v["tickets"]) > 10 else ""
            out.append(f"| {v.get('kind', '')} | {v['formation']} | {v['ranker']} | {v['condition']} | {tix} | {_pct(v['roi'])} | "
                       f"{_pct(v['roi_first'])} / {_pct(v['roi_second'])} | {_pct(v['ci_low'])}〜{_pct(v['ci_high'])} | "
                       f"{v['races']:,} |")
        out.append("\n「堅実」は回収率がほぼ確定している買い方（多くは 100% 未満）、「狙い目候補」は過去は 100% を"
                   "超えたが少数の高配当頼みでまだ確かめられていない買い方です。"
                   "偶然の幅の下限が 100% を下回るものは、プラスと言い切れません。\n")
    if kind == "real" and res["drawn"]:
        pass
    elif kind == "yoso":
        good = rows[rows["ev"] >= min_ev]
        names = "、".join(f"{x['horse_name']}({x['ev']:.2f})" for _, x in good.iterrows()) or "なし"
        out.append(f"### 予想オッズ時点で単勝の期待値が {min_ev:.2f} 以上の馬\n\n{names}\n")
        out.append("実際のオッズが出たら再計算します。現時点では購入額は出しません。\n")
    else:
        out.append("オッズ発表後に買い目を計算します。それまでは「買える下限オッズ」を目安にしてください。\n")
    return "\n".join(out)


def to_json(res: dict) -> dict:
    rows = res["rows"]
    return {
        "race": res["card"]["race"], "odds_kind": res["kind"], "odds_time": res["card"].get("odds_time"),
        "fetched_at": res["card"].get("fetched_at"),
        "runners": [{"number": None if pd.isna(x["number"]) else int(x["number"]), "horse": x["horse_name"],
                     "horse_id": x["horse_id"], "p": round(float(x["p"]), 4),
                     "p_fundamental": round(float(x["p_fund"]), 4),
                     "win_odds": None if pd.isna(x["odds"]) else float(x["odds"]),
                     "ev": None if pd.isna(x["ev"]) else round(float(x["ev"]), 3)} for _, x in rows.iterrows()],
        "validated_strategies": res.get("validated", []),
        "bets": [{"type": b.bet_type, "combo": list(b.combo), "prob": round(b.prob, 5), "odds": b.odds,
                  "ev": round(b.ev, 3), "stake": b.stake} for b in res["bets"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race-ids", default="")
    ap.add_argument("--bankroll", type=int, default=100_000)
    ap.add_argument("--min-ev", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=5000, help="yen per race for the budget slip (0 = none)")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args(argv)
    ids = [x for x in args.race_ids.split(",") if x]
    cards = load_cards(ids or None, since=today_jst())
    if not cards:
        print("no upcoming race cards in data/cards (run: python -m keiba.cards)")
        return 0
    art = None if args.retrain else pipeline.load_artifacts()
    if art is None:
        res = pipeline.fit_all()
        art = res["art"]
        pipeline.save_artifacts(art)
        pipeline.write_report(res)
        from . import strategy
        strategy.write_strategy_report(res)
    boosters = pipeline.boosters(art)
    races, runners, ped = cards_to_frames(cards)
    df = features.build_features(card_races=races, card_runners=runners, pedigree_extra=ped)
    missing = [f for f in art["feats"] if f not in df.columns]
    if missing:   # a restored model from older code: train a fresh one rather than fail
        print(f"model/feature mismatch ({missing[:5]}), retraining")
        res = pipeline.fit_all()
        art = res["art"]
        pipeline.save_artifacts(art)
        boosters = pipeline.boosters(art)
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for c in sorted(cards, key=lambda c: (c["race"]["date"], c["race"].get("start_time") or "", c["race"]["race_id"])):
        rid = c["race"]["race_id"]
        rows = df[df["race_id"] == rid]
        if len(rows) < 2:
            continue
        res = predict_race(c, rows, art, boosters, args.bankroll, args.min_ev, args.budget)
        text = render(res, art, args.bankroll, args.min_ev)
        stem = f"{c['race']['date'].replace('-', '')}_{rid}"
        (OUT / f"{stem}.md").write_text(text)
        (OUT / f"{stem}.json").write_text(json.dumps(to_json(res), ensure_ascii=False, indent=1))
        sections.append(text)
        print(text)
    header = (f"# JRA 予想（{today_jst().isoformat()} 更新）\n\n"
              "期待値は「推定的中確率 × オッズ」で、1.00 が損益ゼロです。"
              f"買い目は期待値 {args.min_ev:.2f} 以上のものを挙げています。"
              "過去の検証では、この選び方は回収率を上げますが、100% を確実に超えるとは言えていません。"
              "買い方ごとの成績は [strategy.md](strategy.md) と [backtest.md](backtest.md)。\n\n")
    (ROOT / "reports" / "latest.md").write_text(header + "\n---\n\n".join(sections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
