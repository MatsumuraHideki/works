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

from . import betting, exotic, features, pipeline, strategy

ROOT = Path(__file__).resolve().parent.parent


RECENT_YEARS = 4     # every row also reports the last few seasons on their own


def _row(group: str, label: str, how: str, stake: np.ndarray, ret: np.ndarray, hits: int, races: int,
         universe: str = "", hit_basis: str = "点", recent: tuple | None = None) -> dict:
    """``recent`` = (first year, stake per race, return per race) of the latest seasons."""
    r3 = lambda v: None if v is None or np.isnan(v) else round(float(v), 4)
    lo, hi = exotic.boot_ci(stake.astype(float), ret.astype(float))
    n = int(stake.sum())
    row = {"group": group, "label": label, "how": how, "races": int(races), "tickets": n, "universe": universe,
           "hit_basis": hit_basis,
           "hit_rate": round(hits / max(n, 1), 4), "roi": round(float(ret.sum() / max(n, 1)), 4),
           "ci_low": r3(lo), "ci_high": r3(hi),
           "recent_from": None, "races_recent": None, "roi_recent": None, "recent_low": None, "recent_high": None}
    if recent is not None and len(recent[1]) and np.sum(recent[1]) > 0:
        y0, s, r = recent[0], np.asarray(recent[1], dtype=float), np.asarray(recent[2], dtype=float)
        lo, hi = exotic.boot_ci(s, r)
        row.update({"recent_from": int(y0), "races_recent": len(s), "roi_recent": r3(r.sum() / s.sum()),
                    "recent_low": r3(lo), "recent_high": r3(hi)})
    return row


def _by_race(d: pd.DataFrame, mask: pd.Series, col: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    s = d[mask]
    g = s.groupby("race_id")[col].agg(["size", "sum"])
    return g["size"].to_numpy(), g["sum"].to_numpy(), int((s[col] > 0).sum()), len(g)


def _recent(d: pd.DataFrame, mask: pd.Series, col: str, y0: int) -> tuple:
    g = d[mask & (d["date"].dt.year >= y0)].groupby("race_id")[col].agg(["size", "sum"])
    return y0, g["size"].to_numpy(), g["sum"].to_numpy()


def _win_frame(oos: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """Test-period runners with the official 単勝 return per 1 staked and the model EV."""
    pw = payouts[payouts["bet_type"] == "単勝"].copy()
    pw["number"] = pd.to_numeric(pw["combo"], errors="coerce")
    pw = pw.dropna(subset=["number", "payout"]).groupby(["race_id", "number"], as_index=False)["payout"].sum()
    o = oos.merge(pw, on=["race_id", "number"], how="left")
    o["ret_win"] = o["payout"].fillna(0.0) / 100.0          # dead heats pay what was actually paid
    o["ev"] = o["p_final"] * o["win_odds"]
    return o


def odds_sensitivity(res: dict, raw: Path = features.RAW, sigmas=(0.0, 0.05, 0.10, 0.20),
                     thresholds=(1.00, 1.10), n_sim: int = 20) -> list[dict]:
    """What the 単勝 rules return if the odds seen when buying differ from the final odds.

    The odds at purchase are simulated as final odds x exp(sigma * z), the bet is chosen on
    those, and it is paid at the final odds. The win probability is left as estimated, and
    the noise is unbiased, so this is a rough guide: real late money is not random.
    """
    _, _, payouts = features.load_raw(raw)
    o = _win_frame(res["oos"], payouts)
    o = o[o["p_final"] >= 0.01]
    p, odds, ret = o["p_final"].to_numpy(), o["win_odds"].to_numpy(dtype=float), o["ret_win"].to_numpy()
    rng = np.random.default_rng(0)
    out = []
    for thr in thresholds:
        for sg in sigmas:
            st = rt = 0.0
            for _ in range(1 if sg == 0 else n_sim):
                sel = p * odds * np.exp(sg * rng.standard_normal(len(o))) >= thr
                st += sel.sum(); rt += ret[sel].sum()
            out.append({"threshold": thr, "sigma": sg, "roi": round(float(rt / max(st, 1)), 4)})
    return out


GRADED = ("G1", "G2", "G3")


def build(res: dict, raw: Path = features.RAW, budget: int = 5000) -> list[dict]:
    art, oos = res["art"], res["oos"]
    races, results, payouts = features.load_raw(raw)
    d = strategy.per_horse_returns(features.build_frame(races, results), payouts)
    span = f"{d['date'].dt.year.min()}〜{d['date'].dt.year.max()}年の全レース"
    test = f"{oos['date'].dt.year.min()}〜{oos['date'].dt.year.max()}年の全レース（検証期間）"
    y0 = int(oos["date"].dt.year.max()) - RECENT_YEARS + 1
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
        rows.append(_row("基準" if "全頭" in label else "人気で選ぶ", label, how, *_by_race(d, mask, col),
                         universe=span, recent=_recent(d, mask, col, y0)))
    # --- 単勝 by model expected value (walk-forward test races), settled with the official payout
    o = _win_frame(oos, payouts)
    pick = (o["p_final"] >= 0.01)
    for thr in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10):
        m = (o["ev"] >= thr) & pick
        rows.append(_row("モデルの期待値で選ぶ", f"単勝 期待値{thr:.2f}以上", "モデル+オッズ",
                         *_by_race(o, m, "ret_win"), universe=test, recent=_recent(o, m, "ret_win", y0)))
    m = (o["ev"] >= 1.0) & (o["ev"] < 1.1) & pick
    rows.append(_row("モデルの期待値で選ぶ", "単勝 期待値1.00以上1.10未満", "モデル+オッズ",
                     *_by_race(o, m, "ret_win"), universe=test, recent=_recent(o, m, "ret_win", y0)))
    is_graded, is_g1 = o["grade"].isin(GRADED), o["grade"].eq("G1")
    for thr in (1.00, betting.WIN_SLIP_EV):
        for sub_mask, name, uni in ((is_graded, "重賞だけ", "重賞（G1〜G3）"), (is_g1, "G1だけ", "G1")):
            m = (o["ev"] >= thr) & pick & sub_mask
            rows.append(_row("モデルの期待値で選ぶ", f"単勝 期待値{thr:.2f}以上（{name}）", "モデル+オッズ",
                             *_by_race(o, m, "ret_win"), universe=uni, recent=_recent(o, m, "ret_win", y0)))
    # the model's top pick of a race, backed only when its own win EV reaches the threshold
    top = o.loc[o.groupby("race_id")["p_final"].idxmax()]
    for thr in (1.00, 1.05):
        rows.append(_row("モデルの期待値で選ぶ", f"単勝 モデル1位の馬（その期待値が{thr:.2f}以上のレースだけ）",
                         "モデル+オッズ", *_by_race(top, top["ev"] >= thr, "ret_win"), universe=test,
                         recent=_recent(top, top["ev"] >= thr, "ret_win", y0)))
    # --- multi-horse pools (open-class races, the only ones with a full odds table)
    odds_df = features.load_odds(raw)
    if len(odds_df):
        tables = pipeline.odds_tables(odds_df)
        pay = strategy.payout_maps(payouts[payouts["race_id"].isin(tables)])
        ex = exotic.build(oos, tables, pay, art["lam2"], art["lam3"])
        subsets = {"オープン以上": set(ex), "重賞（G1〜G3）": {r for r, t in ex.items() if t["grade"] in GRADED},
                   "G1": {r for r, t in ex.items() if t["grade"] == "G1"}}

        def pool(types, thr, label, group, how, universe="オープン以上", floor=0.005):
            st, rt, hits, late = [], [], 0, []
            for rid in subsets[universe]:
                tabs = ex[rid]
                n = r = 0.0
                for bt in types:
                    t = tabs.get(bt)
                    if t is None:
                        continue
                    sel = (t["p"] >= floor) & ((t["p"] * t["odds"] >= thr) if thr else True)
                    n += sel.sum(); r += t["ret"][sel].sum(); hits += int((t["ret"][sel] > 0).sum())
                if n:
                    st.append(n); rt.append(r); late.append(tabs["date"].year >= y0)
            if st:
                st, rt, late = np.array(st), np.array(rt), np.array(late)
                rows.append(_row(group, label, how, st, rt, hits, len(st), universe=universe,
                                 recent=(y0, st[late], rt[late])))

        pool(exotic.CORE, None, "馬連・ワイド・三連複を全通り買う", "基準", "選ばずに全部", floor=0.0)
        pool(exotic.CORE, None, "馬連・ワイド・三連複（モデルが的中確率0.5%以上と見た買い目を全部）",
             "モデルの期待値で選ぶ", "モデル+オッズ")
        for thr in (0.90, 1.00, 1.15):
            pool(exotic.CORE, thr, f"馬連・ワイド・三連複 期待値{thr:.2f}以上", "モデルの期待値で選ぶ", "モデル+オッズ")
        pool(exotic.CORE, 1.00, "馬連・ワイド・三連複 期待値1.00以上（重賞だけ）", "モデルの期待値で選ぶ", "モデル+オッズ",
             universe="重賞（G1〜G3）")
        pool(exotic.CORE, 1.00, "馬連・ワイド・三連複 期待値1.00以上（G1だけ）", "モデルの期待値で選ぶ", "モデル+オッズ",
             universe="G1")
        for bt in exotic.CORE:
            pool([bt], 1.00, f"{bt} 期待値1.00以上", "モデルの期待値で選ぶ", "モデル+オッズ")

        # --- the two budget slips exactly as predictions publish them
        tops = top.set_index("race_id")["number"].astype(int).to_dict()
        for universe, ids in subsets.items():
            acc = {"A": ([], [], 0, []), "B": ([], [], 0, [])}
            for rid in ids:
                tabs = ex[rid]
                cand, rets = [], {}
                for bt in ("単勝",) + betting.CORE_TYPES:
                    t = tabs.get(bt)
                    if t is None:
                        continue
                    for k, p, od, r in zip(t["keys"], t["p"], t["odds"], t["ret"]):
                        cand.append(betting.Bet(bt, k, float(p), float(od), float(p * od), 0.0))
                        rets[(bt, k)] = r
                win_slip, slip = betting.make_slips(cand, tops.get(rid, -1), budget)
                for name, sl in (("A", win_slip), ("B", slip)):
                    if not sl:
                        continue
                    st, rt, hits, late = acc[name]
                    got = sum(b.stake * rets.get((b.bet_type, b.combo), 0.0) for b in sl)
                    st.append(sum(b.stake for b in sl) / 100.0); rt.append(got / 100.0)
                    late.append(tabs["date"].year >= y0)
                    acc[name] = (st, rt, hits + int(got > 0), late)
            for name, text in (("A", "買い目A（単勝だけ）"), ("B", "買い目B（単勝＋馬連・ワイド・三連複）")):
                st, rt, hits, late = acc[name]
                if st:
                    st, rt, late = np.array(st), np.array(rt), np.array(late)
                    r = _row("予想レポートの買い目", f"{text} {universe}", "モデル+オッズ", st, rt,
                             0, len(st), universe=universe, hit_basis="レース", recent=(y0, st[late], rt[late]))
                    r["hit_rate"] = round(hits / len(st), 4)        # share of races with any winning ticket
                    r["tickets"] = None
                    rows.append(r)
    # --- fixed formations that held up best
    top_path = ROOT / "reports" / "strategy_top.json"
    if top_path.exists():
        for t in json.loads(top_path.read_text()):
            if t.get("kind") != "堅実":
                continue
            rows.append({"group": "決まった形で買う", "label": f"{t['formation']}（{t['condition']}）",
                         "how": t["ranker"], "races": t["races"], "tickets": None, "hit_rate": round(t["hit"], 4),
                         "hit_basis": "レース", "universe": test,
                         "roi": round(t["roi"], 4), "ci_low": round(t["ci_low"], 4), "ci_high": round(t["ci_high"], 4),
                         "recent_from": None, "races_recent": None, "roi_recent": None,
                         "recent_low": None, "recent_high": None})
    for i, r in enumerate(sorted(rows, key=lambda r: -r["roi"])):
        r["rank"] = i + 1
    rows.sort(key=lambda r: r["rank"])
    return rows


def write_markdown(rows: list[dict], sens: list[dict] | None = None) -> str:
    """The same table as a page that reads on a phone (reports/summary.md)."""
    pc = lambda v: "—" if v is None else f"{v * 100:.1f}%"
    by = {r["label"]: r for r in rows}
    y0 = next((r["recent_from"] for r in rows if r.get("recent_from")), None)

    def line(label: str, note: str = "") -> str | None:
        r = by.get(label)
        if r is None:
            return None
        rec = f"、{y0}年以降は {pc(r['roi_recent'])}" if r.get("roi_recent") is not None else ""
        return f"- {label}: **{pc(r['roi'])}**（偶然の幅 {pc(r['ci_low'])}〜{pc(r['ci_high'])}{rec}）{note}"

    thr = betting.WIN_SLIP_EV
    out = ["# 買い方ごとの回収率\n",
           "回収率 = 払戻 ÷ 購入額。100% で損益ゼロ。公式の払戻金で計算。"
           "「偶然の幅」は、たまたまの当たり外れで数字がぶれる範囲（95%）で、"
           "幅が 100% をまたぐ買い方は得か損かをまだ言い切れません。"
           "モデルを使う買い方は、各年をそれより前のレースだけで学習したモデルで予測した結果です。\n",
           "## 先に結論\n"]
    out += [x for x in (
        line("単勝を全頭買う", " … 何も選ばないとこの水準"),
        line(f"単勝 期待値{thr:.2f}以上", " … 予想レポートの買い目 A の基準。"
             + ("偶然の幅ごと 100% を超えている" if (by.get(f"単勝 期待値{thr:.2f}以上") or {}).get("ci_low", 0) >= 1.0
                else "100% は超えたが、偶然でも説明できる範囲") + "。年ごとの振れが大きい"),
        line("単勝 期待値1.00以上1.10未満", " … 損益ゼロ前後。買ったあとにオッズが下がるとマイナス"),
        line("馬連・ワイド・三連複 期待値1.00以上", " … 選ばないよりずっと良いが、プラスとは言えない"),
        line("馬連・ワイド・三連複を全通り買う"),
    ) if x]
    out.append("\nつまり、回収率を上げる買い方は「単勝で、モデルの期待値が十分に高い馬だけを買い、"
               "該当馬がいないレースは見送る」です。何も選ばない場合より 30 ポイント以上高くなりますが、"
               "下の表のとおり、買ったあとのオッズの動きを入れると損益ゼロ付近で、確実にプラスになる買い方は見つかっていません。"
               "馬連・ワイド・三連複は当たれば大きい代わりに、長く買うと 100% 前後かそれ以下です。\n")
    if sens:
        out += ["## 買ったあとにオッズが動いたら\n",
                "検証は確定オッズですが、実際に買うのは締切前です。買った時点のオッズが確定オッズから"
                "ランダムにずれていたとして計算し直すと、単勝の回収率はこうなります"
                "（勝率の見積もりは据え置きの簡易計算。実際の締切前の買いは強い馬に集まりやすいので、これより悪くなりえます）。\n",
                "| オッズのずれ | " + " | ".join(f"期待値{t:.2f}以上" for t in sorted({x["threshold"] for x in sens})) + " |",
                "|---|" + "---|" * len({x["threshold"] for x in sens})]
        for sg in sorted({x["sigma"] for x in sens}):
            cells = [pc(x["roi"]) for x in sorted(sens, key=lambda x: x["threshold"]) if x["sigma"] == sg]
            out.append(f"| {'なし（確定オッズ）' if sg == 0 else f'約{sg * 100:.0f}%'} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## 一覧（回収率の高い順）\n",
            "**読むときの注意**\n",
            "- 行によって対象レースが違います（「対象」の列）。馬連・ワイド・三連複は全通りのオッズがあるオープン以上のレースだけです。",
            "- 期待値の基準はいくつも試したうえで並べています。上位の行は、その分だけ実力より良く見えています。",
            f"- 「{y0}年〜」は直近 {RECENT_YEARS} 年だけの回収率です。全期間と大きく違う行は、昔だけ良かった（または悪かった）買い方です。",
            "- 的中率は「点」が 1 点あたり、「レース」が 1 点でも当たったレースの割合です。\n",
            f"| 順位 | 買い方 | 回収率 | 偶然の幅 | {y0}年〜 | 的中率 | レース数 | 対象 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['rank']} | {r['label']} | **{pc(r['roi'])}** | {pc(r['ci_low'])}〜{pc(r['ci_high'])} | "
                   f"{pc(r.get('roi_recent'))} | {pc(r['hit_rate'])}（{r.get('hit_basis', '点')}） | {r['races']:,} | "
                   f"{r.get('universe', '')} |")
    out.append("\n詳しい内訳は [strategy.md](strategy.md)（買い方別）と [backtest.md](backtest.md)（予測精度）。\n")
    text = "\n".join(out)
    (ROOT / "reports" / "summary.md").write_text(text)
    return text


def write_all(res: dict) -> list[dict]:
    rows = build(res)
    sens = odds_sensitivity(res)
    (ROOT / "reports" / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (ROOT / "reports" / "odds_sensitivity.json").write_text(json.dumps(sens, ensure_ascii=False, indent=1))
    write_markdown(rows, sens)
    return rows


def main() -> int:
    res = pipeline.fit_all(log=lambda *a: None)
    rows = write_all(res)
    for r in rows:
        ci = f"{r['ci_low'] * 100:5.1f}〜{r['ci_high'] * 100:5.1f}%" if r["ci_low"] is not None else "     -"
        print(f"{r['rank']:>2} {r['roi'] * 100:6.1f}% ({ci}) {r['group']} | {r['label']} | {r['races']:,} races")
    return 0


if __name__ == "__main__":
    sys.exit(main())
