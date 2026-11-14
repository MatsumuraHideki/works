"""Which way of betting returns the most? Return-rate (回収率) analysis of betting strategies.

Three layers, all settled with the official payouts:

A. Market facts (no model, every race in the data): return by bet type, by
   popularity of the horse / of the ticket, by odds band, by race condition.
B. Formation strategies on walk-forward test races: fixed ticket shapes built from
   the model's ranking or from the market's ranking, optionally filtered by race
   conditions. The model never saw the races it is scored on.
C. Robustness: every strategy is scored on an earlier "discovery" part and a later
   "confirmation" part, with a bootstrap confidence interval, because the best of
   many strategies always looks better in hindsight than it is.

Usage:
    python -m keiba.strategy            # trains (walk-forward) and writes reports/strategy.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, pipeline

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]
TAKEOUT = {"単勝": 80.0, "複勝": 80.0, "枠連": 77.5, "馬連": 77.5, "ワイド": 77.5, "馬単": 75.0,
           "三連複": 75.0, "三連単": 72.5}


# ------------------------------------------------------------------ helpers

def payout_maps(payouts: pd.DataFrame) -> dict[tuple[str, str], dict[tuple[int, ...], float]]:
    """{(race_id, bet_type): {combo: payout per 100 yen}}"""
    out: dict[tuple[str, str], dict[tuple[int, ...], float]] = {}
    p = payouts.dropna(subset=["payout"])
    for rid, bt, combo, pay in zip(p["race_id"], p["bet_type"], p["combo"], p["payout"]):
        out.setdefault((rid, bt), {})[pipeline.parse_combo(bt, combo)] = float(pay)
    return out


def boot_ci(stake: np.ndarray, ret: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the return rate, resampling races."""
    n = len(stake)
    if n < 20 or stake.sum() == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    s, r = stake[idx].sum(axis=1), ret[idx].sum(axis=1)
    roi = r / np.where(s > 0, s, np.nan)
    return float(np.nanpercentile(roi, 2.5)), float(np.nanpercentile(roi, 97.5))


def _pct(x: float) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def md_table(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "(該当なし)\n"
    d = df.copy()
    for c in d.columns:
        if "率" in c or "下限" in c or "上限" in c:
            d[c] = [_pct(v) for v in d[c]]
        elif d[c].dtype.kind == "f":
            d[c] = [("" if pd.isna(v) else f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}") for v in d[c]]
        elif d[c].dtype.kind in "iu":
            d[c] = [f"{v:,}" for v in d[c]]
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ A. market facts

def per_horse_returns(df: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """Runner-level frame with the 単勝 / 複勝 return of a 100-yen bet on that runner."""
    d = df[(~df["is_card"]) & df["win_odds"].notna() & df["number"].notna()].copy()
    for bt, col in (("単勝", "ret_win"), ("複勝", "ret_place")):
        p = payouts[payouts["bet_type"] == bt].copy()
        p["number"] = pd.to_numeric(p["combo"], errors="coerce")
        p = p.dropna(subset=["number", "payout"]).groupby(["race_id", "number"], as_index=False)["payout"].sum()
        d = d.merge(p.rename(columns={"payout": col}), on=["race_id", "number"], how="left")
        d[col] = d[col].fillna(0.0) / 100.0
    has_place = d.groupby("race_id")["ret_place"].transform("sum") > 0
    d["place_sold"] = has_place
    d["odds_band"] = pd.cut(d["win_odds"], [0, 1.9, 2.9, 4.9, 9.9, 19.9, 49.9, 99.9, 1e9],
                            labels=["1.0-1.9", "2.0-2.9", "3.0-4.9", "5.0-9.9", "10-19.9", "20-49.9", "50-99.9", "100以上"])
    d["pop_band"] = pd.cut(d["popularity"], [0, 1, 2, 3, 5, 8, 12, 99],
                           labels=["1番人気", "2番人気", "3番人気", "4-5番人気", "6-8番人気", "9-12番人気", "13番人気以下"])
    d["grade_grp"] = np.where(d["grade"].isin(["G1", "G2", "G3"]), d["grade"],
                              np.where(d["class"].eq("OP"), "OP・L", d["class"]))
    d["dist_band"] = pd.cut(d["distance"], [0, 1300, 1650, 2050, 2500, 9999],
                            labels=["〜1300m", "1400-1600m", "1700-2000m", "2100-2500m", "2600m〜"])
    d["field_band"] = pd.cut(d["n_runners"], [0, 9, 13, 16, 99], labels=["〜9頭", "10-13頭", "14-16頭", "17頭〜"])
    return d


def single_table(d: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    g = d.groupby(by, observed=True)
    t = g.agg(頭数=("ret_win", "size"), 勝率=("win", "mean"), 単勝回収率=("ret_win", "mean")).reset_index()
    pl = d[d["place_sold"]].groupby(by, observed=True).agg(
        複勝的中率=("ret_place", lambda s: (s > 0).mean()), 複勝回収率=("ret_place", "mean")).reset_index()
    return t.merge(pl, on=by, how="left").rename(columns={by: label})


def n_tickets(bet_type: str, n: np.ndarray) -> np.ndarray:
    """How many different tickets of a bet type exist in a race with n starters."""
    n = n.astype(float)
    if bet_type in ("単勝", "複勝"):
        return n
    if bet_type in ("馬連", "ワイド"):
        return n * (n - 1) / 2
    if bet_type == "馬単":
        return n * (n - 1)
    if bet_type == "三連複":
        return n * (n - 1) * (n - 2) / 6
    if bet_type == "三連単":
        return n * (n - 1) * (n - 2)
    b = np.minimum(n, 8)                       # 枠連: bracket pairs + same-bracket pairs
    return b * (b - 1) / 2 + np.clip(n - 8, 0, 8)


def ticket_popularity_table(payouts: pd.DataFrame, runners: pd.Series) -> pd.DataFrame:
    """Return rate of always buying the k-th most popular ticket of each bet type.

    ``runners``: starters per race (index race_id), used to count the tickets that exist.
    """
    bands = [(1, 1, "1番人気"), (2, 3, "2-3番人気"), (4, 6, "4-6番人気"), (7, 12, "7-12番人気"),
             (13, 30, "13-30番人気"), (31, 100, "31-100番人気"), (101, 10**7, "101番人気以下")]
    p = payouts.dropna(subset=["payout", "popularity"])
    rows = []
    for bt in TYPES:
        q = p[p["bet_type"] == bt]
        if not len(q):
            continue
        n = runners.reindex(q["race_id"].unique()).dropna().to_numpy()
        total = n_tickets(bt, n)
        row = {"券種": bt, "理論上の回収率": TAKEOUT[bt] / 100}
        for lo, hi, name in bands:
            stake = np.clip(np.minimum(hi, total) - lo + 1, 0, None).sum()
            sel = q[(q["popularity"] >= lo) & (q["popularity"] <= hi)]
            row[f"{name}の回収率"] = sel["payout"].sum() / (100.0 * stake) if stake >= 500 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def condition_scan(d: pd.DataFrame, min_n: int = 300) -> pd.DataFrame:
    """単勝 / 複勝 return for popularity band x one race condition, split in two halves by date."""
    dates = np.sort(d["date"].unique())
    cut = dates[len(dates) // 2]
    conds = {"クラス": "grade_grp", "芝ダ": "surface", "距離": "dist_band", "頭数": "field_band",
             "馬場": "going", "競馬場": "place"}
    rows = []
    for cname, col in conds.items():
        for (cval, pop), g in d.groupby([col, "pop_band"], observed=True):
            if len(g) < min_n:
                continue
            a, b = g[g["date"] < cut], g[g["date"] >= cut]
            for bt, rc in (("単勝", "ret_win"), ("複勝", "ret_place")):
                gg = g if bt == "単勝" else g[g["place_sold"]]
                if len(gg) < min_n:
                    continue
                lo, hi = boot_ci(np.ones(len(gg)), gg[rc].to_numpy())
                rows.append({"券種": bt, "条件": f"{cname}={cval}", "人気": pop, "点数": len(gg),
                             "回収率": gg[rc].mean(), "前半の回収率": a[rc].mean() if len(a) else np.nan,
                             "後半の回収率": b[rc].mean() if len(b) else np.nan, "95%下限": lo, "95%上限": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. formation strategies

@dataclass
class Formation:
    name: str
    bet_type: str
    k: int                      # how many top-ranked horses it needs

    def tickets(self, top: list[int]) -> list[tuple[int, ...]]:
        n = self.name
        if n in ("単勝 1位", "複勝 1位"):
            return [(top[0],)]
        if n == "複勝 1-2位":
            return [(top[0],), (top[1],)]
        if n in ("馬連 1-2位", "ワイド 1-2位"):
            return [tuple(sorted(top[:2]))]
        if n in ("馬連 上位3頭ボックス", "ワイド 上位3頭ボックス"):
            return [tuple(sorted(c)) for c in combinations(top[:3], 2)]
        if n in ("馬連 1位軸-2〜5位", "ワイド 1位軸-2〜5位"):
            return [tuple(sorted((top[0], x))) for x in top[1:5]]
        if n == "馬単 1位→2位":
            return [(top[0], top[1])]
        if n == "馬単 1位→2〜4位":
            return [(top[0], x) for x in top[1:4]]
        if n == "三連複 1-2-3位":
            return [tuple(sorted(top[:3]))]
        if n == "三連複 上位4頭ボックス":
            return [tuple(sorted(c)) for c in combinations(top[:4], 3)]
        if n == "三連複 上位5頭ボックス":
            return [tuple(sorted(c)) for c in combinations(top[:5], 3)]
        if n == "三連複 1位軸-2〜6位":
            return [tuple(sorted((top[0],) + c)) for c in combinations(top[1:6], 2)]
        if n == "三連単 1位→2位→3位":
            return [tuple(top[:3])]
        if n == "三連単 上位3頭ボックス":
            return list(permutations(top[:3], 3))
        if n == "三連単 1位1着固定-2〜4位":
            return [(top[0],) + c for c in permutations(top[1:4], 2)]
        raise ValueError(n)


FORMATIONS = [
    Formation("単勝 1位", "単勝", 1), Formation("複勝 1位", "複勝", 1), Formation("複勝 1-2位", "複勝", 2),
    Formation("馬連 1-2位", "馬連", 2), Formation("馬連 上位3頭ボックス", "馬連", 3),
    Formation("馬連 1位軸-2〜5位", "馬連", 5),
    Formation("ワイド 1-2位", "ワイド", 2), Formation("ワイド 上位3頭ボックス", "ワイド", 3),
    Formation("ワイド 1位軸-2〜5位", "ワイド", 5),
    Formation("馬単 1位→2位", "馬単", 2), Formation("馬単 1位→2〜4位", "馬単", 4),
    Formation("三連複 1-2-3位", "三連複", 3), Formation("三連複 上位4頭ボックス", "三連複", 4),
    Formation("三連複 上位5頭ボックス", "三連複", 5), Formation("三連複 1位軸-2〜6位", "三連複", 6),
    Formation("三連単 1位→2位→3位", "三連単", 3), Formation("三連単 上位3頭ボックス", "三連単", 3),
    Formation("三連単 1位1着固定-2〜4位", "三連単", 4),
]

RANKERS = {"モデル+オッズ": "p_final", "モデル単独": "p_fund", "人気順": "p_market"}


def race_filters(race: pd.DataFrame) -> dict[str, pd.Series]:
    """Race-level conditions (index = race_id) under which a formation is played."""
    f = {"全レース": pd.Series(True, index=race.index)}
    f["重賞(G1-G3)"] = race["grade"].isin(["G1", "G2", "G3"])
    f["OP・L・重賞"] = race["class"].eq("OP")
    f["条件戦(1勝〜3勝)"] = race["class"].isin(["1勝", "2勝", "3勝"])
    f["未勝利・新馬"] = race["class"].isin(["未勝利", "新馬"])
    f["芝"] = race["surface"].eq("芝")
    f["ダート"] = race["surface"].eq("ダ")
    f["少頭数(〜11頭)"] = race["n_runners"] <= 11
    f["多頭数(16頭〜)"] = race["n_runners"] >= 16
    f["1番人気が1倍台"] = race["fav_odds"] < 2.0
    f["1番人気が3倍以上"] = race["fav_odds"] >= 3.0
    f["モデル1位≠1番人気"] = race["model_top"] != race["fav"]
    f["モデル1位=1番人気"] = race["model_top"] == race["fav"]
    f["モデル1位の勝率30%以上"] = race["model_top_p"] >= 0.30
    f["モデル1位の単勝期待値1.0以上"] = race["model_top_ev"] >= 1.0
    f["モデル単独1位が4番人気以下"] = race["fund_top_pop"] >= 4
    return f


def race_attrs(rid: str, g: pd.DataFrame) -> dict:
    """Race-level facts the filters look at. ``g``: runners with p_final, p_fund, p_market, win_odds."""
    nums = g["number"].astype(int).to_numpy()
    pm = g["p_market"].to_numpy(dtype=float)
    has_odds = bool(np.isfinite(pm).all())
    fav_i = int(np.argmax(pm)) if has_odds else 0
    top_i = int(np.argmax(g["p_final"].to_numpy()))
    fund_i = int(np.argmax(g["p_fund"].to_numpy()))
    odds = g["win_odds"].to_numpy(dtype=float)
    pop = pd.Series(odds).rank(method="min").to_numpy() if has_odds else np.full(len(g), np.nan)
    first = g.iloc[0]
    return {"race_id": rid, "date": first["date"], "grade": first["grade"], "class": first["class"],
            "surface": first["surface"], "n_runners": len(g),
            "fav": int(nums[fav_i]) if has_odds else -1,
            "fav_odds": float(odds[fav_i]) if has_odds else np.nan, "model_top": int(nums[top_i]),
            "model_top_p": float(g["p_final"].iloc[top_i]),
            "model_top_ev": float(g["p_final"].iloc[top_i] * odds[top_i]) if has_odds else np.nan,
            "fund_top_pop": float(pop[fund_i])}


def formation_results(oos: pd.DataFrame, pay: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stake and return of every (ranker, formation) on every test race.

    Returns (per-race table of stakes/returns, race-level attribute frame).
    """
    recs = []
    race_rows = []
    for rid, g in oos.groupby("race_id", sort=False):
        g = g.dropna(subset=["number"])
        if len(g) < 5:
            continue
        nums = g["number"].astype(int).to_numpy()
        order = {name: nums[np.argsort(-g[col].to_numpy(), kind="stable")] for name, col in RANKERS.items()}
        race_rows.append(race_attrs(rid, g))
        for rk, top in order.items():
            top = [int(x) for x in top]
            for fm in FORMATIONS:
                table = pay.get((rid, fm.bet_type))
                if table is None or len(top) < fm.k:
                    continue
                tix = fm.tickets(top)
                ret = sum(table.get(t, 0.0) for t in tix) / 100.0
                recs.append((rid, rk, fm.name, fm.bet_type, len(tix), ret))
    res = pd.DataFrame(recs, columns=["race_id", "ranker", "formation", "bet_type", "stake", "ret"])
    return res, pd.DataFrame(race_rows).set_index("race_id")


def summarise(res: pd.DataFrame, race: pd.DataFrame, min_races: int = 150) -> pd.DataFrame:
    filt = race_filters(race)
    dates = np.sort(race["date"].unique())
    cut = dates[int(len(dates) * 0.5)]
    early = set(race.index[race["date"] < cut])
    rows = []
    for (rk, fname, bt), g in res.groupby(["ranker", "formation", "bet_type"], sort=False):
        g = g.set_index("race_id")
        for cname, mask in filt.items():
            if rk == "人気順" and cname.startswith("モデル"):
                continue
            ids = mask.index[mask.fillna(False)]
            s = g[g.index.isin(ids)]
            if len(s) < min_races:
                continue
            stake, ret = s["stake"].to_numpy(dtype=float), s["ret"].to_numpy()
            is_early = s.index.isin(early)
            lo, hi = boot_ci(stake, ret)
            rows.append({
                "選び方": rk, "買い方": fname, "条件": cname, "券種": bt, "レース数": len(s),
                "点数": int(stake.sum()), "的中率": float((ret > 0).mean()), "回収率": float(ret.sum() / stake.sum()),
                "前半の回収率": float(ret[is_early].sum() / max(stake[is_early].sum(), 1)),
                "後半の回収率": float(ret[~is_early].sum() / max(stake[~is_early].sum(), 1)),
                "95%下限": lo, "95%上限": hi,
            })
    return pd.DataFrame(rows)


def top_strategies(summ: pd.DataFrame, n: int = 4) -> list[dict]:
    """Strategies to carry into live predictions, in two honest groups:

    堅実      highest 95% lower bound: the return is known fairly precisely (usually < 100%).
    狙い目候補  at least 100% in both the earlier and the later half, ranked by the worse
              half. These rest on few big payouts, so the interval is wide and unproven.
    """
    if not len(summ):
        return []
    s = summ.copy()
    s["weak"] = s[["前半の回収率", "後半の回収率"]].min(axis=1)
    solid = s.sort_values("95%下限", ascending=False).drop_duplicates(["買い方"]).head(n).assign(kind="堅実")
    cand = s[(s["weak"] >= 1.0) & ~s.index.isin(solid.index)]
    cand = cand.sort_values(["weak", "95%下限"], ascending=False).drop_duplicates(["買い方"]).head(n).assign(kind="狙い目候補")
    return [{"kind": r["kind"], "ranker": r["選び方"], "formation": r["買い方"], "condition": r["条件"],
             "races": int(r["レース数"]), "hit": float(r["的中率"]), "roi": float(r["回収率"]),
             "roi_first": float(r["前半の回収率"]), "roi_second": float(r["後半の回収率"]),
             "ci_low": float(r["95%下限"]), "ci_high": float(r["95%上限"])}
            for _, r in pd.concat([solid, cand]).iterrows()]


def apply_strategies(strategies: list[dict], g: pd.DataFrame, rid: str) -> list[dict]:
    """Tickets each stored strategy would buy in one (upcoming) race, if its condition holds."""
    if not strategies or g["number"].isna().any() or len(g) < 5:
        return []
    race = pd.DataFrame([race_attrs(rid, g)]).set_index("race_id")
    filt = race_filters(race)
    nums = g["number"].astype(int).to_numpy()
    forms = {f.name: f for f in FORMATIONS}
    out = []
    for st in strategies:
        col = RANKERS[st["ranker"]]
        if g[col].isna().any() or st["condition"] not in filt or not bool(filt[st["condition"]].fillna(False).iloc[0]):
            continue
        fm = forms[st["formation"]]
        top = [int(x) for x in nums[np.argsort(-g[col].to_numpy(), kind="stable")]]
        if len(top) < fm.k:
            continue
        out.append({**st, "bet_type": fm.bet_type, "tickets": [list(t) for t in fm.tickets(top)]})
    return out


# ------------------------------------------------------------------ report

def write_strategy_report(res: dict, raw: Path = features.RAW, log=print) -> str:
    art, oos = res["art"], res["oos"]
    races, results, payouts = features.load_raw(raw)
    df = features.build_frame(races, results)
    d = per_horse_returns(df, payouts)
    n_races = d["race_id"].nunique()
    out = ["# どう買えば回収率が高いか（過去データの分析）\n"]
    out.append(f"- 対象: {d['date'].min().date()}〜{d['date'].max().date()} の平地 {n_races:,} レース（新馬〜G1）")
    out.append("- 回収率 = 払戻合計 ÷ 購入額合計。100% で損益ゼロ。すべて 100 円均等買い、公式の払戻金で計算。")
    out.append("- 「95%下限〜上限」は、たまたまの当たり外れでどのくらい数字がぶれるかの幅です。"
               "下限が 100% を超えていない限り、プラスだとは言えません。\n")

    out.append("## 1. 券種ごとの基本（人気順に買い続けた場合）\n")
    out.append("各レースで「その券種の中で k 番目に売れている買い目」を 1 点ずつ買い続けた場合の回収率です。\n")
    runners = d.groupby("race_id")["n_runners"].first()
    out.append(md_table(ticket_popularity_table(payouts[payouts["race_id"].isin(set(d["race_id"]))], runners)))
    out.append("空欄は、その人気帯の買い目がほとんど存在しない券種です。\n")

    out.append("## 2. 単勝・複勝：人気別／オッズ別\n")
    out.append(md_table(single_table(d, "pop_band", "人気")))
    out.append(md_table(single_table(d, "odds_band", "単勝オッズ")))

    out.append("## 3. 単勝・複勝：レース条件 × 人気（回収率が高い組み合わせ）\n")
    scan = condition_scan(d, min_n=max(150, int(n_races * 0.03)))
    if len(scan):
        best = scan.sort_values("回収率", ascending=False).head(15)
        out.append("回収率の上位 15 件（点数が一定以上あるものだけ）。前半と後半の両方で高いものだけが信用できます。\n")
        out.append(md_table(best))
        worst = scan.sort_values("回収率").head(8)
        out.append("逆に回収率が低い（避けたい）組み合わせ:\n")
        out.append(md_table(worst))

    summ = pd.DataFrame()
    if len(oos):
        pay = payout_maps(payouts[payouts["race_id"].isin(set(oos["race_id"]))])
        fr, race = formation_results(oos, pay)
        summ = summarise(fr, race, min_races=max(60, int(len(race) * 0.04)))
        fm = art["fold_months"]
        out.append("## 4. 買い方（フォーメーション）別の回収率\n")
        out.append(f"検証レース {len(race):,}（{race['date'].min().date()}〜{race['date'].max().date()}）。"
                   f"モデルは各期間をそれより前のデータだけで学習（{fm}か月区切り）。"
                   "「選び方」は上位馬を何の順位で決めるか（モデル+オッズ／モデル単独／人気順）です。\n")
        base = summ[summ["条件"] == "全レース"].sort_values(["券種", "買い方", "選び方"])
        cols = ["買い方", "選び方", "レース数", "点数", "的中率", "回収率", "95%下限", "95%上限"]
        out.append("### 全レースで買い続けた場合\n")
        out.append(md_table(base[cols]))

        out.append("### 条件を絞った場合の回収率 上位 20\n")
        out.append("多数の組み合わせから良いものを拾っているので、上位は実力以上に良く見えます。"
                   "前半・後半の両方で 100% を超え、95%下限も高いものを重視してください。\n")
        cols2 = ["選び方", "買い方", "条件", "レース数", "点数", "的中率", "回収率", "前半の回収率", "後半の回収率",
                 "95%下限", "95%上限"]
        out.append(md_table(summ.sort_values("回収率", ascending=False).head(20)[cols2]))

        robust = summ[(summ["前半の回収率"] >= 1.0) & (summ["後半の回収率"] >= 1.0)].sort_values("95%下限", ascending=False)
        out.append("### 前半・後半ともに 100% 以上だった買い方\n")
        out.append(md_table(robust.head(15)[cols2]) if len(robust) else "該当なし。\n")

        out.append("### モデルは人気順より良い選び方か\n")
        piv = base.pivot_table(index="買い方", columns="選び方", values="回収率", sort=False).reset_index()
        piv.columns = [c if c == "買い方" else f"{c}の回収率" for c in piv.columns]
        out.append(md_table(piv))

    out.append("## まとめ\n")
    out.extend(conclusions(d, payouts, summ))
    tops = top_strategies(summ)
    if tops:
        out.append("## 今後の予想に当てはめる買い方\n")
        out.append("- **堅実**: 偶然の幅の下限が最も高い買い方。回収率がほぼ確定していて、大きくは負けにくい。\n"
                   "- **狙い目候補**: 前半・後半とも 100% 以上だった買い方。少数の高配当に支えられていて幅が広く、"
                   "本当にプラスかはまだ分からない。\n\n"
                   "予想レポートでは、条件に当てはまるレースでこれらの買い目を表示します。\n")
        out.append(md_table(pd.DataFrame([{"区分": t["kind"], "買い方": t["formation"], "選び方": t["ranker"], "条件": t["condition"],
                                            "レース数": t["races"], "的中率": t["hit"], "回収率": t["roi"], "前半の回収率": t["roi_first"],
                                            "後半の回収率": t["roi_second"], "95%下限": t["ci_low"],
                                            "95%上限": t["ci_high"]} for t in tops])))
    (REPORTS / "strategy_top.json").parent.mkdir(parents=True, exist_ok=True)
    (REPORTS / "strategy_top.json").write_text(json.dumps(tops, ensure_ascii=False, indent=1))
    text = "\n".join(out)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "strategy.md").write_text(text)
    log(f"wrote {REPORTS / 'strategy.md'}")
    return text


def conclusions(d: pd.DataFrame, payouts: pd.DataFrame, summ: pd.DataFrame) -> list[str]:
    """Plain-language findings computed from the tables (nothing hard-coded)."""
    out = []
    pop = single_table(d, "pop_band", "人気")
    bw, bp = pop.loc[pop["単勝回収率"].idxmax()], pop.loc[pop["複勝回収率"].idxmax()]
    ww = pop.loc[pop["単勝回収率"].idxmin()]
    out.append(f"- 人気別では、単勝は **{bw['人気']}**（回収率 {_pct(bw['単勝回収率'])}）、複勝は **{bp['人気']}**"
               f"（{_pct(bp['複勝回収率'])}）が最も高く、単勝の **{ww['人気']}**（{_pct(ww['単勝回収率'])}）が最も低い。")
    if len(summ):
        base = summ[summ["条件"] == "全レース"]
        if len(base):
            b = base.loc[base["回収率"].idxmax()]
            out.append(f"- 全レースで買い続ける形では **{b['買い方']}（{b['選び方']}）** が最高で回収率 {_pct(b['回収率'])}"
                       f"（95%幅 {_pct(b['95%下限'])}〜{_pct(b['95%上限'])}、{b['レース数']:,} レース）。")
        t = summ.loc[summ["回収率"].idxmax()]
        out.append(f"- 条件を絞った中の最高は **{t['買い方']}（{t['選び方']}／{t['条件']}）** の {_pct(t['回収率'])}"
                   f"（前半 {_pct(t['前半の回収率'])}、後半 {_pct(t['後半の回収率'])}、95%下限 {_pct(t['95%下限'])}）。")
        n_pos = int((summ["95%下限"] > 1.0).sum())
        out.append(f"- 調べた {len(summ):,} 通りのうち、95%下限が 100% を超えた（偶然では説明しにくい）ものは **{n_pos} 通り**。")
        n_both = int(((summ["前半の回収率"] >= 1.0) & (summ["後半の回収率"] >= 1.0)).sum())
        out.append(f"- 前半・後半の両方で 100% 以上だったものは {n_both} 通り。")
    out.append("- 控除率（単複 20%、馬連・ワイド 22.5%、馬単・三連複 25%、三連単 27.5%）の分、"
               "何も工夫しなければ券種ごとの理論値に近づきます。理論値をどれだけ上回れるかが工夫の効果です。\n")
    return out


def main(argv: list[str] | None = None) -> int:
    res = pipeline.fit_all()
    pipeline.save_artifacts(res["art"])
    pipeline.write_report(res)
    print(write_strategy_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
