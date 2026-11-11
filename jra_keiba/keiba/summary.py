"""One comparable table of ways to bet and what each returned (reports/summary.json).

Every row is settled with official payouts at 100 yen per ticket and carries a
95% bootstrap interval (resampling races), so rows can be compared directly.

Usage:
    python -m keiba.summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import exotic, features, pipeline, strategy

ROOT = Path(__file__).resolve().parent.parent


def _row(group: str, label: str, how: str, stake: np.ndarray, ret: np.ndarray, hits: int, races: int) -> dict:
    lo, hi = exotic.boot_ci(stake.astype(float), ret.astype(float))
    n = int(stake.sum())
    return {"group": group, "label": label, "how": how, "races": int(races), "tickets": n,
            "hit_rate": round(hits / max(n, 1), 4), "roi": round(float(ret.sum() / max(n, 1)), 4),
            "ci_low": None if np.isnan(lo) else round(lo, 4), "ci_high": None if np.isnan(hi) else round(hi, 4)}


def _by_race(d: pd.DataFrame, mask: pd.Series, col: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    s = d[mask]
    g = s.groupby("race_id")[col].agg(["size", "sum"])
    return g["size"].to_numpy(), g["sum"].to_numpy(), int((s[col] > 0).sum()), len(g)


def build(res: dict, raw: Path = features.RAW) -> list[dict]:
    art, oos = res["art"], res["oos"]
    races, results, payouts = features.load_raw(raw)
    d = strategy.per_horse_returns(features.build_frame(races, results), payouts)
    rows = []
    # --- no selection / by popularity (every race in the data)
    for label, how, mask, col in [
        ("単勝を全頭買う", "選ばずに全部", d["win_odds"] > 0, "ret_win"),
        ("単勝 1番人気", "人気だけで選ぶ", d["popularity"] == 1, "ret_win"),
        ("複勝 1番人気", "人気だけで選ぶ", (d["popularity"] == 1) & d["place_sold"], "ret_place"),
        ("複勝 単勝1倍台の馬", "人気だけで選ぶ", (d["win_odds"] < 2.0) & d["place_sold"], "ret_place"),
        ("単勝 13番人気以下", "人気だけで選ぶ", d["popularity"] >= 13, "ret_win"),
        ("単勝 100倍以上", "人気だけで選ぶ", d["win_odds"] >= 100, "ret_win"),
    ]:
        rows.append(_row("基準" if "全頭" in label else "人気で選ぶ", label, how, *_by_race(d, mask, col)))
    # --- 単勝 by model expected value (walk-forward test races)
    o = oos.assign(ret_win=oos["win"] * oos["win_odds"], ev=oos["p_final"] * oos["win_odds"])
    for thr in (0.85, 0.90, 0.95):
        rows.append(_row("モデルの期待値で選ぶ", f"単勝 期待値{thr:.2f}以上", "モデル+オッズ",
                         *_by_race(o, (o["ev"] >= thr) & (o["p_final"] >= 0.01), "ret_win")))
    # --- multi-horse pools by model expected value (races with full odds)
    odds_df = features.load_odds(raw)
    if len(odds_df):
        tables = pipeline.odds_tables(odds_df)
        pay = strategy.payout_maps(payouts[payouts["race_id"].isin(tables)])
        ex = exotic.build(oos, tables, pay, art["lam2"], art["lam3"])

        def pool(types, thr, label, group, how):
            st, rt, hits = [], [], 0
            for tabs in ex.values():
                n = r = 0.0
                for bt in types:
                    t = tabs.get(bt)
                    if t is None:
                        continue
                    sel = (t["p"] >= 0.005) & ((t["p"] * t["odds"] >= thr) if thr else True)
                    n += sel.sum(); r += t["ret"][sel].sum(); hits += int((t["ret"][sel] > 0).sum())
                if n:
                    st.append(n); rt.append(r)
            if st:
                rows.append(_row(group, label, how, np.array(st), np.array(rt), hits, len(st)))

        pool(exotic.CORE, None, "馬連・ワイド・三連複を選ばず買う", "基準", "選ばずに全部")
        for thr in (0.90, 1.00, 1.15):
            pool(exotic.CORE, thr, f"馬連・ワイド・三連複 期待値{thr:.2f}以上", "モデルの期待値で選ぶ", "モデル+オッズ")
        for bt in exotic.CORE:
            pool([bt], 1.00, f"{bt} 期待値1.00以上", "モデルの期待値で選ぶ", "モデル+オッズ")
        pool(["三連単"], None, "三連単を選ばず買う", "避けたい買い方", "選ばずに全部")
        pool(["三連単"], 1.00, "三連単 期待値1.00以上", "避けたい買い方", "モデル+オッズ")
    # --- fixed formations that held up best
    top = ROOT / "reports" / "strategy_top.json"
    if top.exists():
        for t in json.loads(top.read_text()):
            if t.get("kind") != "堅実":
                continue
            rows.append({"group": "決まった形で買う", "label": f"{t['formation']}（{t['condition']}）",
                         "how": t["ranker"], "races": t["races"], "tickets": None, "hit_rate": round(t["hit"], 4),
                         "roi": round(t["roi"], 4), "ci_low": round(t["ci_low"], 4), "ci_high": round(t["ci_high"], 4)})
    for i, r in enumerate(sorted(rows, key=lambda r: -r["roi"])):
        r["rank"] = i + 1
    rows.sort(key=lambda r: r["rank"])
    return rows


def main() -> int:
    res = pipeline.fit_all(log=lambda *a: None)
    rows = build(res)
    out = ROOT / "reports" / "summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        ci = f"{r['ci_low'] * 100:5.1f}〜{r['ci_high'] * 100:5.1f}%" if r["ci_low"] is not None else "     -"
        print(f"{r['rank']:>2} {r['roi'] * 100:6.1f}% ({ci}) {r['group']} | {r['label']} | {r['races']:,} races")
    return 0


if __name__ == "__main__":
    sys.exit(main())
