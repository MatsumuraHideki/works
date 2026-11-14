"""Check every published prediction against the actual result.

For each reports/predictions/<date>_<race_id>.json (or .md for slips published before the
slip was stored in the JSON) whose result is known, settle the budget slip ticket by
ticket and keep a running record in reports/track_record.md / .json.

Results come from data/cards/results/<race_id>.json (race-day result page) or, once the
weekly collection has run, from data/raw/payouts.

Usage:
    python -m keiba.settle
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

from . import features
from .pipeline import ORDERED

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "reports" / "predictions"
LIVE = ROOT / "data" / "cards" / "results"
OUT = ROOT / "reports"
JST = dt.timezone(dt.timedelta(hours=9))


def _key(bet_type: str, combo) -> tuple[int, ...]:
    c = tuple(int(x) for x in combo)
    return c if bet_type in ORDERED or bet_type in ("単勝", "複勝") else tuple(sorted(c))


def slip_from_markdown(text: str) -> list[dict]:
    """Slips written before the JSON carried them: read the 予算 table of the report."""
    m = re.search(r"### 予算 [\d,]+ 円の買い目(?: B[^\n]*)?\n(.*?)(?:\*\*まとめて買う場合|\n合計 )", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[2].endswith("円") or cells[0] == "券種":
            continue
        combo = [int(x) for x in re.split(r"[-→]", cells[1]) if x.isdigit()]
        out.append({"type": cells[0], "combo": combo, "stake": int(cells[2].rstrip("円").replace(",", "")),
                    "odds": float(cells[3])})
    return out


def load_result(race_id: str, raw_payouts: pd.DataFrame | None) -> dict | None:
    p = LIVE / f"{race_id}.json"
    if p.exists():
        r = json.loads(p.read_text())
        pay = {(x["bet_type"], _key(x["bet_type"], x["combo"])): x["payout"] / 100.0 for x in r["payouts"]}
        top3 = [f"{int(o['number'])} {o['horse_name']}" for o in r["order"][:3] if o.get("number")]
        return {"pay": pay, "top3": top3, "source": "当日の結果ページ"}
    if raw_payouts is not None and len(raw_payouts):
        g = raw_payouts[raw_payouts["race_id"] == race_id]
        if len(g):
            pay = {}
            for bt, combo, v in zip(g["bet_type"], g["combo"], g["payout"]):
                nums = [x for x in str(combo).split("-") if x.strip().isdigit()]
                pay[(bt, _key(bt, nums))] = float(v) / 100.0
            return {"pay": pay, "top3": [], "source": "蓄積データ"}
    return None


def settle_all(log=print) -> list[dict]:
    _, _, raw_payouts = features.load_raw()
    rows = []
    for jp in sorted(PRED.glob("*.json")):
        d = json.loads(jp.read_text())
        race = d["race"]
        slip = d.get("slip") or []
        if not slip and jp.with_suffix(".md").exists():
            slip = slip_from_markdown(jp.with_suffix(".md").read_text())
        res = load_result(race["race_id"], raw_payouts)
        # a "prediction" whose snapshot was taken after post time must never be scored
        try:
            post = dt.datetime.fromisoformat(f"{race['date']}T{race['start_time']}:00").replace(tzinfo=JST)
            snap = dt.datetime.fromisoformat(str(d.get("fetched_at")))
            snap = snap if snap.tzinfo else snap.replace(tzinfo=JST)
            if snap >= post:
                log(f"skip {race['race_id']}: snapshot {snap} is not before post time {post}")
                continue
        except (KeyError, TypeError, ValueError):
            pass
        win_slip = d.get("win_slip") or []
        row = {"date": race["date"], "race_id": race["race_id"], "race": race["name"], "grade": race.get("grade"),
               "odds_time": d.get("odds_time"), "tickets": len(slip), "stake": sum(t["stake"] for t in slip),
               "settled": res is not None}
        if res is not None and slip:
            hits = []
            ret = 0.0
            for t in slip:
                pay = res["pay"].get((t["type"], _key(t["type"], t["combo"])), 0.0)
                if pay > 0:
                    got = pay * t["stake"]
                    ret += got
                    sep = "→" if t["type"] in ORDERED else "-"
                    hits.append(f"{t['type']} {sep.join(map(str, t['combo']))}（{t['stake']:,}円 → {got:,.0f}円）")
            a_ret = sum(res["pay"].get((t["type"], _key(t["type"], t["combo"])), 0.0) * t["stake"] for t in win_slip)
            row.update({"a_tickets": len(win_slip), "a_stake": sum(t["stake"] for t in win_slip), "a_return": round(a_ret),
                        "a_picks": [f"{t['combo'][0]}番 {t['stake']:,}円" for t in win_slip]})
            top = d["runners"][0] if d.get("runners") else None
            row.update({"return": round(ret), "hits": hits, "top3": res["top3"], "source": res["source"],
                        "model_top": f"{top['number']} {top['horse']}" if top else None})
        rows.append(row)
    return rows


def write_record(rows: list[dict]) -> str:
    done = [r for r in rows if r["settled"] and r["tickets"]]
    out = ["# 予想の結果（毎回の答え合わせ）\n",
           "予想レポートで出した「予算の買い目」を、実際の払戻金で 1 点ずつ精算した記録です。\n"]
    if done:
        stake, ret = sum(r["stake"] for r in done), sum(r["return"] for r in done)
        out.append(f"**通算: {len(done)} レース、購入 {stake:,} 円、払戻 {ret:,} 円、回収率 {ret / stake * 100:.1f}%、"
                   f"当たりがあったレース {sum(1 for r in done if r['return'] > 0)} / {len(done)}**\n")
        a = [r for r in done if r.get("a_stake")]
        if a:
            sa, ra = sum(r["a_stake"] for r in a), sum(r["a_return"] for r in a)
            out.append(f"**買い目 A（単勝だけ）の通算: {len(a)} レース、購入 {sa:,} 円、払戻 {ra:,} 円、回収率 {ra / sa * 100:.1f}%**\n")
            out.append("| 日付 | レース | A の買い目 | 購入 | 払戻 |")
            out.append("|---|---|---|---|---|")
            for r in sorted(a, key=lambda r: (r["date"], r["race_id"]), reverse=True):
                out.append(f"| {r['date']} | {r['race']} | {'、'.join(r['a_picks'])} | {r['a_stake']:,}円 | {r['a_return']:,}円 |")
            out.append("")
        out.append("買い目 B（単勝＋馬連・ワイド・三連複）:\n")
        out.append("| 日付 | レース | 点数 | 購入 | 払戻 | 収支 | 当たった買い目 | 1〜3着 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(done, key=lambda r: (r["date"], r["race_id"]), reverse=True):
            grade = f"（{r['grade']}）" if r.get("grade") else ""
            out.append(f"| {r['date']} | {r['race']}{grade} | {r['tickets']} | {r['stake']:,}円 | {r['return']:,}円 | "
                       f"{r['return'] - r['stake']:+,}円 | {'、'.join(r['hits']) or 'なし'} | {' / '.join(r['top3']) or '-'} |")
        out.append("")
    pending = [r for r in rows if not r["settled"] and r["tickets"]]
    if pending:
        out.append("結果待ち: " + "、".join(f"{r['date']} {r['race']}" for r in pending) + "\n")
    out.append("数レースの収支は運の振れが大きく、買い方の良し悪しはこの表だけでは判断できません。"
               "長期の成績は [summary.md](summary.md) の検証結果を見てください。\n")
    text = "\n".join(out)
    (OUT / "track_record.md").write_text(text)
    (OUT / "track_record.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return text


def main() -> int:
    print(write_record(settle_all()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
