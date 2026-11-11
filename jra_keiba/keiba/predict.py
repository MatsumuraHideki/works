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

from . import betting, features, model, pipeline
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


def mark(values: pd.Series) -> pd.Series:
    """Within-race marks for an aspect score: ◎ best, ○ top quarter, △ above median."""
    r = values.rank(ascending=False, method="min")
    n = len(values)
    return pd.Series(np.where(r == 1, "◎", np.where(r <= max(2, n * 0.25), "○",
                     np.where(r <= n * 0.5, "△", "－"))), index=values.index)


def predict_race(card: dict, rows: pd.DataFrame, art: dict, boosters: dict, bankroll: int, min_ev: float
                 ) -> dict:
    feats = art["feats"]
    rows = rows.copy()
    s = model.predict_stage1(boosters, rows, feats)
    rows["s_win"], rows["s_top3"] = s["s_win"], s["s_top3"]
    rows["p_fund"] = model.Stage2(["s_win", "s_top3"], np.asarray(art["s2_fund"])).predict(rows)
    tables, win_odds, kind = card_odds(card, rows)
    rows["odds"] = win_odds
    use_market = kind != "none" and rows["odds"].notna().all()
    if use_market:
        inv = 1.0 / rows["odds"]
        rows["m"] = np.log(inv / inv.sum())
        rows["p"] = model.Stage2(["s_win", "s_top3", "m"], np.asarray(art["s2_full"])).predict(rows)
    else:
        rows["p"] = rows["p_fund"]
    rows["ev"] = rows["p"] * rows["odds"]
    contrib = model.group_contributions(boosters["win"], rows[feats], art["groups"])
    for g in ASPECTS:
        rows[f"a_{g}"] = mark(contrib[g]) if g in contrib else "－"

    drawn = rows["number"].notna().all()
    bets: list[betting.Bet] = []
    top_by_type: dict[str, list[betting.Bet]] = {}
    if kind == "real" and drawn and len(rows) >= 2:
        probs = betting.bet_probabilities(rows["p"].to_numpy(), [int(x) for x in rows["number"]],
                                          art["lam2"], art["lam3"])
        cand = betting.evaluate_bets(probs, tables)
        bets = betting.size_bets(betting.select_bets(cand, min_ev=min_ev), bankroll=bankroll)
        for bt in betting.BET_TYPES:
            top = sorted((b for b in cand if b.bet_type == bt and b.prob >= 0.005),
                         key=lambda b: b.ev, reverse=True)[:5]
            if top:
                top_by_type[bt] = top
    return {"card": card, "rows": rows.sort_values("p", ascending=False), "kind": kind,
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

    if kind == "real" and res["drawn"]:
        out.append(f"### 推奨買い目（資金 {bankroll:,} 円、期待値 {min_ev:.2f} 以上、1/4ケリー配分）\n")
        if res["bets"]:
            out.append("| 券種 | 買い目 | 的中確率 | オッズ | 期待値 | 購入額 |")
            out.append("|---|---|---|---|---|---|")
            for b in res["bets"]:
                out.append(f"| {b.bet_type} | {b.label()} | {_pct(b.prob)} | {b.odds:.1f} | {b.ev:.2f} | {b.stake:,}円 |")
            stake, ret = betting.portfolio_ev(res["bets"])
            out.append(f"\n合計 {stake:,} 円、期待払戻 {ret:,.0f} 円（期待回収率 {ret / stake * 100:.0f}%）\n")
        else:
            out.append("期待値の基準を満たす買い目なし。このレースは**見送り**。\n")
        out.append("### 券種別の期待値上位\n")
        out.append("| 券種 | 買い目 | 的中確率 | オッズ | 期待値 |")
        out.append("|---|---|---|---|---|")
        for bt, top in res["top_by_type"].items():
            for b in top[:3]:
                out.append(f"| {bt} | {b.label()} | {_pct(b.prob)} | {b.odds:.1f} | {b.ev:.2f} |")
        out.append("")
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
        "bets": [{"type": b.bet_type, "combo": list(b.combo), "prob": round(b.prob, 5), "odds": b.odds,
                  "ev": round(b.ev, 3), "stake": b.stake} for b in res["bets"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race-ids", default="")
    ap.add_argument("--bankroll", type=int, default=100_000)
    ap.add_argument("--min-ev", type=float, default=1.15)
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
    if missing:
        raise SystemExit(f"model/feature mismatch, retrain with --retrain: {missing[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for c in sorted(cards, key=lambda c: (c["race"]["date"], c["race"].get("start_time") or "", c["race"]["race_id"])):
        rid = c["race"]["race_id"]
        rows = df[df["race_id"] == rid]
        if len(rows) < 2:
            continue
        res = predict_race(c, rows, art, boosters, args.bankroll, args.min_ev)
        text = render(res, art, args.bankroll, args.min_ev)
        stem = f"{c['race']['date'].replace('-', '')}_{rid}"
        (OUT / f"{stem}.md").write_text(text)
        (OUT / f"{stem}.json").write_text(json.dumps(to_json(res), ensure_ascii=False, indent=1))
        sections.append(text)
        print(text)
    header = (f"# JRA 予想（{today_jst().isoformat()} 更新）\n\n"
              "期待値は「推定的中確率 × オッズ」。1.00 が損益ゼロで、モデルの誤差を見込んで "
              f"{args.min_ev:.2f} 以上だけを推奨しています。検証結果は [backtest.md](backtest.md)。\n\n")
    (ROOT / "reports" / "latest.md").write_text(header + "\n---\n\n".join(sections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
