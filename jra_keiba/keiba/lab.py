"""Experiments: ways of staking, selecting and modelling, tested on the walk-forward predictions.

    python -m keiba.lab            -> reports/experiments.md, reports/experiments.json

Input is the out-of-sample table of the last full fit (data/work/oos.pkl) and, when
present, the experimental heads from ``python -m keiba.lab_heads``.

Every idea is scored three ways: over the whole test period, on the DISCOVERY years
(up to 2020) and on the CONFIRMATION years (2021 on). Ideas are meant to be picked on the
discovery years only; the confirmation column then says whether they held up. With this
many ideas, some will look good by luck, so a row counts only if it also holds in the
confirmation years.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import betting, exotic, features, model, pipeline, strategy, summary

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
SPLIT = 2021                      # first confirmation year
BUDGET, UNIT = 5000, 100


# ------------------------------------------------------------------ helpers

def score(stake: np.ndarray, ret: np.ndarray, year: np.ndarray) -> dict:
    """Return rate with a 95% interval, overall and for the two halves. One entry per race."""
    stake, ret, year = np.asarray(stake, float), np.asarray(ret, float), np.asarray(year)
    out = {}
    for name, m in (("all", np.ones(len(stake), bool)), ("disc", year < SPLIT), ("conf", year >= SPLIT)):
        s, r = stake[m], ret[m]
        lo, hi = exotic.boot_ci(s, r, 600)
        out[name] = {"roi": float(r.sum() / s.sum()) if s.sum() else np.nan, "lo": lo, "hi": hi,
                     "races": int((s > 0).sum()), "hit": float((r[s > 0] > 0).mean()) if (s > 0).any() else np.nan}
    return out


def by_race(d: pd.DataFrame, w: np.ndarray) -> dict:
    """Score a set of 単勝 bets (rows of ``d``) staked ``w`` each."""
    g = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "s": w, "r": w * d["ret_win"].to_numpy(),
                      "y": d["date"].dt.year.to_numpy()}).groupby("race_id").agg(s=("s", "sum"), r=("r", "sum"),
                                                                              y=("y", "first"))
    return score(g["s"], g["r"], g["y"])


def allocate(w: np.ndarray, budget: int = BUDGET, unit: int = UNIT) -> np.ndarray:
    """Split a budget over tickets in proportion to ``w``, in whole units (largest remainder).
    With more tickets than units, the heaviest get one unit each."""
    n_units = budget // unit
    w = np.asarray(w, float)
    out = np.zeros(len(w))
    if not len(w) or w.sum() <= 0:
        return out
    if len(w) > n_units:
        keep = np.argsort(-w, kind="stable")[:n_units]
        out[keep] = unit
        return out
    raw = w / w.sum() * n_units
    base = np.floor(raw)
    left = int(n_units - base.sum())
    if left:
        base[np.argsort(-(raw - base), kind="stable")[:left]] += 1
    return base * unit


# ------------------------------------------------------------------ 1. staking 単勝

WIN_WEIGHTS = {
    "均等（全部同じ額）": lambda p, o: np.ones_like(p),
    "勝率に比例（当たりやすい馬に厚く）": lambda p, o: p,
    "払戻が同額になるように（オッズの逆数）": lambda p, o: 1.0 / o,
    "期待値の上乗せ分に比例（期待値−1）": lambda p, o: np.clip(p * o - 1.0, 1e-9, None),
    "ケリー基準（(期待値−1)÷(オッズ−1)）": lambda p, o: np.clip((p * o - 1.0) / (o - 1.0), 1e-9, None),
    "オッズに比例（穴馬に厚く）": lambda p, o: o,
}


def exp_win_staking(o: pd.DataFrame) -> list[dict]:
    rows = []
    for thr in (1.00, 1.10):
        d = o[(o["ev"] >= thr) & (o["p_final"] >= 0.01)]
        p, od = d["p_final"].to_numpy(), d["win_odds"].to_numpy(float)
        for name, fn in WIN_WEIGHTS.items():
            rows.append({"set": f"単勝 期待値{thr:.2f}以上", "weight": name, **by_race(d, fn(p, od))})
    return rows


# ------------------------------------------------------------------ 2. budget slips

def race_candidates(tabs: dict) -> dict | None:
    """Every priced 単勝 / 馬連 / ワイド / 三連複 ticket of one race as flat arrays."""
    bt, p, od, ret, keys = [], [], [], [], []
    for b in ("単勝",) + betting.CORE_TYPES:
        t = tabs.get(b)
        if t is None:
            continue
        bt += [b] * len(t["keys"]); keys += list(t["keys"])
        p.append(t["p"]); od.append(t["odds"]); ret.append(t["ret"])
    if not p:
        return None
    p, od, ret = np.concatenate(p), np.concatenate(od), np.concatenate(ret)
    return {"bt": np.array(bt), "keys": keys, "p": p, "odds": od, "ret": ret, "ev": p * od}


def _top(mask: np.ndarray, key: np.ndarray, n: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    return idx[np.argsort(-key[idx], kind="stable")[:n]]


def _core(c: dict, thr: float = 1.0, types=betting.CORE_TYPES) -> np.ndarray:
    return np.isin(c["bt"], types) & (c["ev"] >= thr) & (c["p"] >= 0.005)


def _win(c: dict, thr: float) -> np.ndarray:
    return (c["bt"] == "単勝") & (c["ev"] >= thr) & (c["p"] >= 0.01)


SELECTIONS = {
    "単勝だけ（期待値1.10以上）": lambda c, top: np.flatnonzero(_win(c, 1.10)),
    "単勝だけ（期待値1.00以上）": lambda c, top: np.flatnonzero(_win(c, 1.00)),
    "当たりやすい順に50点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["p"], 50),
    "当たりやすい順に20点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["p"], 20),
    "当たりやすい順に10点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["p"], 10),
    "当たりやすい順に5点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["p"], 5),
    "期待値の高い順に20点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["ev"], 20),
    "期待値の高い順に10点": lambda c, top: _top(_core(c) | _win(c, betting.WIN_EV), c["ev"], 10),
    "期待値1.20以上を当たりやすい順に20点": lambda c, top: _top(_core(c, 1.2) | _win(c, 1.2), c["p"], 20),
    "馬連だけ（当たりやすい順に20点）": lambda c, top: _top(_core(c, 1.0, ("馬連",)), c["p"], 20),
    "ワイドだけ（当たりやすい順に20点）": lambda c, top: _top(_core(c, 1.0, ("ワイド",)), c["p"], 20),
    "三連複だけ（当たりやすい順に20点）": lambda c, top: _top(_core(c, 1.0, ("三連複",)), c["p"], 20),
    "単勝＋馬連（当たりやすい順に20点）": lambda c, top: _top(_core(c, 1.0, ("馬連",)) | _win(c, betting.WIN_EV), c["p"], 20),
    "ワイド抜き（単勝・馬連・三連複）当たりやすい順に20点": lambda c, top: _top(
        _core(c, 1.0, ("馬連", "三連複")) | _win(c, betting.WIN_EV), c["p"], 20),
    "ワイド抜き 当たりやすい順に10点": lambda c, top: _top(_core(c, 1.0, ("馬連", "三連複")) | _win(c, betting.WIN_EV), c["p"], 10),
    "ワイド抜き 期待値の高い順に20点": lambda c, top: _top(_core(c, 1.0, ("馬連", "三連複")) | _win(c, betting.WIN_EV), c["ev"], 20),
    "ワイド抜き 期待値の高い順に10点": lambda c, top: _top(_core(c, 1.0, ("馬連", "三連複")) | _win(c, betting.WIN_EV), c["ev"], 10),
    "ワイド抜き 期待値の高い順に5点": lambda c, top: _top(_core(c, 1.0, ("馬連", "三連複")) | _win(c, betting.WIN_EV), c["ev"], 5),
    "単勝だけ（期待値1.00以上・5倍未満）": lambda c, top: np.flatnonzero(_win(c, 1.00) & (c["odds"] < 5)),
    "単勝だけ（期待値1.00以上・10倍未満）": lambda c, top: np.flatnonzero(_win(c, 1.00) & (c["odds"] < 10)),
    "モデル1位の馬を含む買い目だけ（20点）": lambda c, top: _top(
        (_core(c) | _win(c, betting.WIN_EV)) & np.array([top in k for k in c["keys"]]), c["p"], 20),
    "モデル1位の馬を含まない買い目だけ（20点）": lambda c, top: _top(
        (_core(c) | _win(c, betting.WIN_EV)) & ~np.array([top in k for k in c["keys"]]), c["p"], 20),
}
SLIP_WEIGHTS = {
    "均等": lambda p, o: np.ones_like(p),
    "的中確率に比例": lambda p, o: p,
    "払戻が同額": lambda p, o: 1.0 / o,
    "ケリー基準": lambda p, o: np.clip((p * o - 1.0) / (o - 1.0), 1e-9, None),
    "期待値の上乗せ分": lambda p, o: np.clip(p * o - 1.0, 1e-9, None),
}


def exp_slips(ex: dict, tops: dict) -> list[dict]:
    """5,000 yen per race: which tickets, and how the money is split between them."""
    cands = {rid: (race_candidates(t), t["date"].year) for rid, t in ex.items()}
    rows = []
    for sname, sel in SELECTIONS.items():
        picked = {rid: sel(c, tops.get(rid, -1)) for rid, (c, _) in cands.items() if c is not None}
        for wname, wf in SLIP_WEIGHTS.items():
            st, rt, yr = [], [], []
            for rid, idx in picked.items():
                if not len(idx):
                    continue
                c, y = cands[rid]
                stake = allocate(wf(c["p"][idx], c["odds"][idx]))
                st.append(stake.sum() / 100.0); rt.append(float((stake * c["ret"][idx]).sum()) / 100.0); yr.append(y)
            if st:
                rows.append({"set": sname, "weight": wname, **score(st, rt, yr)})
    return rows


# ------------------------------------------------------------------ 3. which 単勝 bets

def exp_win_slices(o: pd.DataFrame) -> list[dict]:
    d = o[(o["ev"] >= 1.0) & (o["p_final"] >= 0.01)].copy()
    n = d.groupby("race_id")["race_id"].transform("size")
    field = o.groupby("race_id")["race_id"].transform("size").reindex(d.index)
    rank = o.groupby("race_id")["p_final"].rank(ascending=False, method="first").reindex(d.index)
    ev_f = d["p_fund"] * d["win_odds"]
    cls = np.select([d["grade"].isin(["G1", "G2", "G3"]), d["class"].isin(["OP", "3勝"]),
                     d["class"].isin(["1勝", "2勝"])], ["重賞", "3勝・オープン", "1勝・2勝"], "新馬・未勝利")
    slices = {
        "オッズ": pd.cut(d["win_odds"], [0, 5, 10, 20, 50, 1e9], labels=["5倍未満", "5〜10倍", "10〜20倍", "20〜50倍", "50倍以上"]),
        "人気": pd.cut(d["popularity"], [0, 1, 3, 6, 9, 99], labels=["1番人気", "2〜3番人気", "4〜6番人気", "7〜9番人気", "10番人気以下"]),
        "モデルの順位": pd.cut(rank, [0, 1, 3, 6, 99], labels=["1位", "2〜3位", "4〜6位", "7位以下"]),
        "頭数": pd.cut(field, [0, 10, 14, 99], labels=["10頭以下", "11〜14頭", "15頭以上"]),
        "コース": d["surface"].map({"芝": "芝", "ダ": "ダート"}),
        "クラス": pd.Series(cls, index=d.index),
        "馬場": d["going"].map({"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}),
        "オッズ抜きのモデルでも期待値1.0以上": pd.Series(np.where(ev_f >= 1.0, "はい", "いいえ"), index=d.index),
        "オッズとのズレ補正の向き": pd.Series(np.where(d["s_res"] > 0, "上方修正", "下方修正"), index=d.index),
        "同じレースの該当馬の数": pd.cut(n, [0, 1, 2, 99], labels=["1頭", "2頭", "3頭以上"]),
        "期待値": pd.cut(d["ev"], [0.999, 1.05, 1.10, 1.20, 99], labels=["1.00〜1.05", "1.05〜1.10", "1.10〜1.20", "1.20以上"]),
    }
    rows = []
    for dim, s in slices.items():
        for val in [v for v in pd.unique(s.dropna())]:
            m = (s == val).to_numpy()
            if m.sum() < 300:
                continue
            rows.append({"dim": dim, "value": str(val), "bets": int(m.sum()), **by_race(d[m], np.ones(m.sum()))})
    return rows


def noisy_roi(d: pd.DataFrame, thr: float, cap: float, kelly: bool, sigma: float = 0.10, n_sim: int = 20) -> float:
    """Return rate when the odds seen at purchase are the final odds times exp(sigma * z):
    the bet and its size are decided on the seen odds, the payout is at the final odds."""
    rng = np.random.default_rng(0)
    p, od, ret = d["p_final"].to_numpy(), d["win_odds"].to_numpy(float), d["ret_win"].to_numpy()
    st = rt = 0.0
    for _ in range(n_sim):
        seen = od * np.exp(sigma * rng.standard_normal(len(d)))
        sel = (p * seen >= thr) & (seen < cap) & (p >= 0.01)
        w = np.clip((p * seen - 1.0) / (seen - 1.0), 1e-9, None)[sel] if kelly else np.ones(sel.sum())
        st += w.sum(); rt += (w * ret[sel]).sum()
    return float(rt / st) if st else np.nan


def exp_win_grid(o: pd.DataFrame) -> list[dict]:
    """単勝 by expected-value threshold, odds cap and staking, with the odds-movement stress test."""
    rows = []
    for cap in (3.0, 5.0, 10.0, 20.0, 1e9):
        for thr in (1.00, 1.05, 1.10):
            d = o[(o["ev"] >= thr) & (o["win_odds"] < cap) & (o["p_final"] >= 0.01)]
            if len(d) < 300:
                continue
            p, od = d["p_final"].to_numpy(), d["win_odds"].to_numpy(float)
            for wname, kelly in (("均等", False), ("ケリー基準", True)):
                w = np.clip((p * od - 1.0) / (od - 1.0), 1e-9, None) if kelly else np.ones(len(d))
                yrs = d.assign(w=w, r=w * d["ret_win"]).groupby(d["date"].dt.year).agg(s=("w", "sum"), r=("r", "sum"))
                rows.append({"cap": None if cap > 1e8 else cap, "thr": thr, "weight": wname, "bets": len(d),
                             "noise10": noisy_roi(o, thr, cap, kelly, 0.10), "noise20": noisy_roi(o, thr, cap, kelly, 0.20),
                             "years_above": int(((yrs["r"] / yrs["s"]) > 1).sum()), "years": len(yrs),
                             **by_race(d, w)})
    return rows


KELLY_FULL = 0.05     # Kelly fraction at which a horse gets the whole race budget


def kelly(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.clip((p * o - 1.0) / np.clip(o - 1.0, 1e-9, None), 0.0, None)


WIN_RULES = {
    "期待値1.10以上": lambda p, o: p * o >= 1.10,
    "期待値1.00以上": lambda p, o: p * o >= 1.00,
    "期待値1.00以上で5倍未満": lambda p, o: (p * o >= 1.00) & (o < 5),
    "期待値1.10以上、または 期待値1.00以上で5倍未満": lambda p, o: (p * o >= 1.10) | ((p * o >= 1.00) & (o < 5)),
    "期待値1.10以上、または 期待値1.05以上で5倍未満": lambda p, o: (p * o >= 1.10) | ((p * o >= 1.05) & (o < 5)),
    "期待値1.10以上、または 期待値1.05以上で10倍未満": lambda p, o: (p * o >= 1.10) | ((p * o >= 1.05) & (o < 10)),
}
RULE_WEIGHTS = {
    "均等": lambda p, o: np.ones_like(p),
    "ケリー基準に比例": lambda p, o: np.clip(kelly(p, o), 1e-9, None),
    "ケリー基準（上限つき）": lambda p, o: np.clip(np.minimum(1.0, kelly(p, o) / KELLY_FULL), 0.02, None),
}


def exp_win_rules(o: pd.DataFrame, sigmas=(0.10, 0.20), n_sim: int = 20) -> list[dict]:
    """Candidate 単勝 rules for the published slip, with the odds-movement stress test.

    "ケリー基準（上限つき）" is a stake a person can follow: the race budget times
    min(1, Kelly fraction / KELLY_FULL), at least one unit."""
    d0 = o[o["p_final"] >= 0.01]
    p, od, ret = d0["p_final"].to_numpy(), d0["win_odds"].to_numpy(float), d0["ret_win"].to_numpy()
    rows = []
    for rname, rule in WIN_RULES.items():
        sel = rule(p, od)
        d = d0[sel]
        for wname, wf in RULE_WEIGHTS.items():
            w = wf(p[sel], od[sel])
            yrs = d.assign(w=w, r=w * d["ret_win"]).groupby(d["date"].dt.year).agg(s=("w", "sum"), r=("r", "sum"))
            row = {"rule": rname, "weight": wname, "bets": int(sel.sum()), "races": int(d["race_id"].nunique()),
                   "years_above": int(((yrs["r"] / yrs["s"]) > 1).sum()), "years": len(yrs), **by_race(d, w)}
            rng = np.random.default_rng(0)
            for sg in sigmas:
                st = rt = 0.0
                for _ in range(n_sim):
                    seen = od * np.exp(sg * rng.standard_normal(len(od)))
                    m = rule(p, seen)
                    ww = wf(p[m], seen[m])
                    st += ww.sum(); rt += (ww * ret[m]).sum()
                row[f"noise{int(sg * 100)}"] = float(rt / st) if st else np.nan
            rows.append(row)
    return rows


def exp_combined_rule(o: pd.DataFrame, slices: list[dict], k: int = 3) -> dict:
    """Take the best slice values of the discovery years and bet their union/intersection later."""
    ok = [r for r in slices if r["disc"]["races"] >= 500 and r["dim"] != "期待値"]
    best = sorted(ok, key=lambda r: -r["disc"]["roi"])[:k]
    return {"picked_on_discovery": [{"dim": r["dim"], "value": r["value"], "disc": r["disc"]["roi"],
                                     "conf": r["conf"]["roi"], "conf_lo": r["conf"]["lo"], "conf_hi": r["conf"]["hi"],
                                     "conf_races": r["conf"]["races"]} for r in best]}


# ------------------------------------------------------------------ 4. stage-2 variants

def walk_stage2(o: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Out-of-sample win probabilities from a conditional logit on ``cols`` (year by year)."""
    p = np.full(len(o), np.nan)
    years = sorted(o["year"].unique())
    for y in years[1:]:
        tr, te = o[o["year"] < y], (o["year"] == y).to_numpy()
        p[te] = model.Stage2(cols).fit(tr).predict(o[te])
    return p


def exp_stage2(o: pd.DataFrame, heads: bool) -> list[dict]:
    o = o.copy()
    o["m2"] = o["m"] ** 2
    o["m_field"] = o["m"] * np.log(o.groupby("race_id")["race_id"].transform("size"))
    variants = {
        "現行（勝ち・3着内・オッズ・ズレ補正）": pipeline.FULL_COLS,
        "オッズだけ（人気と不人気の偏りだけ補正）": ["m"],
        "オッズ＋ズレ補正だけ": ["m", "s_res"],
        "ズレ補正なし": ["s_win", "s_top3", "m"],
        "現行＋オッズの2乗": pipeline.FULL_COLS + ["m2"],
        "現行＋オッズ×頭数": pipeline.FULL_COLS + ["m_field"],
    }
    if heads:
        variants.update({
            "現行＋順位学習": pipeline.FULL_COLS + ["s_rank"],
            "勝ちモデルを直近重視に置き換え": ["s_winw", "s_top3", "m", "s_res"],
            "ズレ補正を直近重視に置き換え": ["s_win", "s_top3", "m", "s_resw"],
            "現行＋直近重視のズレ補正": pipeline.FULL_COLS + ["s_resw"],
            "現行＋3着内のズレ補正": pipeline.FULL_COLS + ["s_res3"],
            "全部入り": pipeline.FULL_COLS + ["s_rank", "s_winw", "s_resw", "s_res3"],
        })
    rows = []
    for name, cols in variants.items():
        p = walk_stage2(o, cols)
        ok = np.isfinite(p)
        d = o[ok].assign(p_v=p[ok])
        d["ev_v"] = d["p_v"] * d["win_odds"]
        r = {"variant": name, "cols": cols, "logloss": model.race_logloss(d, d["p_v"].to_numpy())}
        for thr in (1.00, 1.10):
            s = d[(d["ev_v"] >= thr) & (d["p_v"] >= 0.01)]
            r[f"ev{thr:.2f}"] = {"bets": len(s), **by_race(s, np.ones(len(s)))}
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 5. place model for exotics

def fit_place_logit(d: pd.DataFrame, cols: list[str], place: int) -> np.ndarray:
    """Conditional logit for who finishes ``place``-th among the runners not yet placed."""
    d = d[d["finish"].notna() & (d["finish"] >= place)]
    X = d[cols].to_numpy(float)
    codes, n = model._race_index(d["race_id"])
    y = (d["finish"].to_numpy() == place).astype(float)
    has = np.bincount(codes, weights=y, minlength=n)[codes] > 0
    X, y = X[has], y[has]
    codes, n = model._race_index(d["race_id"][has])
    w = y / np.bincount(codes, weights=y, minlength=n)[codes]

    def nll(theta):
        p = model.softmax_by_race(X @ theta, codes, n)
        return (-(w * np.log(np.clip(p, 1e-12, None))).sum() / n + 1e-4 * theta @ theta,
                -(X * (w - p)[:, None]).sum(axis=0) / n + 2e-4 * theta)

    return minimize(nll, np.full(len(cols), 0.3), jac=True, method="L-BFGS-B").x


def exp_place_model(o: pd.DataFrame, tables: dict, pay: dict, lam2: float, lam3: float, heads: bool) -> list[dict]:
    """Second and third place from their own conditional logits instead of p ** lambda."""
    o = o.copy()
    o["lp"] = np.log(np.clip(o["p_final"], 1e-9, None))
    variants = {"現行（勝率のべき乗）": None, "着順ごとのモデル（勝率・3着内・オッズ）": ["lp", "s_top3", "m"]}
    if heads:
        variants["着順ごとのモデル＋3着内のズレ補正"] = ["lp", "s_top3", "m", "s_res3"]
    years = sorted(o["year"].unique())
    rows = []
    for name, cols in variants.items():
        d = o[o["race_id"].isin(tables)].copy()
        if cols is None:
            d["a2"], d["a3"] = d["p_final"] ** lam2, d["p_final"] ** lam3
            d = d[d["year"] >= years[1]]
        else:
            d["a2"] = np.nan; d["a3"] = np.nan
            for y in years[1:]:
                tr = o[o["year"] < y]
                t2, t3 = fit_place_logit(tr, cols, 2), fit_place_logit(tr, cols, 3)
                te = d["year"] == y
                d.loc[te, "a2"] = np.exp(d.loc[te, cols].to_numpy(float) @ t2)
                d.loc[te, "a3"] = np.exp(d.loc[te, cols].to_numpy(float) @ t3)
            d = d.dropna(subset=["a2"])
        ex = exotic.build(d, tables, pay, lam2, lam3, a_col="a2", b_col="a3")
        r = {"variant": name}
        for label, thr in (("期待値1.00以上", 1.0), ("期待値1.15以上", 1.15)):
            st, rt, yr = [], [], []
            for rid, tabs in ex.items():
                n = g = 0.0
                for bt in exotic.CORE:
                    t = tabs.get(bt)
                    if t is None:
                        continue
                    sel = (t["p"] >= 0.005) & (t["p"] * t["odds"] >= thr)
                    n += sel.sum(); g += t["ret"][sel].sum()
                if n:
                    st.append(n); rt.append(g); yr.append(tabs["date"].year)
            r[label] = score(st, rt, yr)
        # how well the ticket probabilities match what happened (higher is better)
        ll, cnt = 0.0, 0
        for rid, tabs in ex.items():
            t = tabs.get("三連複")
            if t is not None and t["ret"].max() > 0:
                ll += float(np.log(np.clip(t["p"][t["ret"] > 0].sum(), 1e-9, None))); cnt += 1
        r["trio_loglik"] = ll / max(cnt, 1)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ report

def _pc(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def _cells(s: dict) -> str:
    a, d, c = s["all"], s["disc"], s["conf"]
    if a["races"] < 100:
        return "該当ほぼなし | — | — | — | —"
    return (f"**{_pc(a['roi'])}** | {_pc(a['lo'])}〜{_pc(a['hi'])} | {_pc(d['roi'])} | {_pc(c['roi'])} | "
            f"{_pc(c['lo'])}〜{_pc(c['hi'])}")


HEAD = f"回収率 | 偶然の幅 | 〜{SPLIT - 1}年 | {SPLIT}年〜 | {SPLIT}年〜の幅"


def write_report(res: dict) -> str:
    out = ["# いろいろ試した結果（賭け金の配分・買い目の選び方・モデル）\n",
           "過去の検証用の予測（各年をそれより前のデータだけで学習）を使い、買い方やモデルの案を同じ物差しで比べました。"
           f"案が多いと、どれかは偶然よく見えます。そこで期間を「〜{SPLIT - 1}年」（案を選ぶ期間）と「{SPLIT}年〜」"
           "（確かめる期間）に分け、**両方で良い案だけを信用する**ことにしています。"
           "回収率は確定オッズ・公式の払戻金で計算しています。\n"]
    if res.get("takeaways"):
        out += ["## わかったこと\n"] + [f"- {t}" for t in res["takeaways"]] + [""]
    out += ["## 1. 単勝の賭け金に重みをつける\n",
            "同じ馬を買うとして、賭け金の配り方だけを変えた場合です。\n",
            f"| 買う馬 | 配り方 | {HEAD} |", "|---|---|---|---|---|---|---|"]
    for r in res.get("win_staking", []):
        out.append(f"| {r['set']} | {r['weight']} | {_cells(r)} |")
    out += ["", "## 2. 1レース5,000円の買い目（オープン以上のレース）\n",
            "「どの買い目を選ぶか」と「5,000円をどう配るか」の組み合わせです。配り方は 100 円単位で、"
            "買い目が 1 点もないレースは買いません。「当たり」は払戻が 1 円でもあったレースの割合です。\n",
            f"| 買い目の選び方 | 配り方 | {HEAD} | 当たり | レース数 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res.get("slips", []), key=lambda r: -r["all"]["roi"]):
        out.append(f"| {r['set']} | {r['weight']} | {_cells(r)} | {_pc(r['all']['hit'])} | {r['all']['races']:,} |")
    out += ["", "## 3. 単勝で期待値1.00以上の馬を、条件で分けると\n",
            f"| 条件 | 区分 | 点数 | {HEAD} |", "|---|---|---|---|---|---|---|---|"]
    for r in res.get("win_slices", []):
        out.append(f"| {r['dim']} | {r['value']} | {r['bets']:,} | {_cells(r)} |")
    cr = res.get("combined_rule", {}).get("picked_on_discovery", [])
    out += ["", f"〜{SPLIT - 1}年で成績が良かった区分の上位と、その後（{SPLIT}年〜）の成績:\n"]
    out += [f"- {r['dim']}＝{r['value']}: 〜{SPLIT - 1}年 {_pc(r['disc'])} → {SPLIT}年〜 {_pc(r['conf'])}"
            f"（幅 {_pc(r['conf_lo'])}〜{_pc(r['conf_hi'])}、{r['conf_races']:,} レース）" for r in cr]
    out += ["", "### 単勝：期待値の基準 × オッズの上限 × 配り方\n",
            "「1割ずれ」「2割ずれ」は、買った時点のオッズが確定オッズからそれだけランダムにずれていた場合の回収率です"
            "（選ぶのも配るのもずれたオッズで行い、払戻は確定オッズ）。「100%超の年」は年ごとに見て 100% を超えた年数です。\n",
            f"| オッズの上限 | 期待値 | 配り方 | 点数 | {HEAD} | 1割ずれ | 2割ずれ | 100%超の年 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("win_grid", []):
        cap = "なし" if r["cap"] is None else f"{r['cap']:.0f}倍未満"
        out.append(f"| {cap} | {r['thr']:.2f}以上 | {r['weight']} | {r['bets']:,} | {_cells(r)} | {_pc(r['noise10'])} | "
                   f"{_pc(r['noise20'])} | {r['years_above']}/{r['years']} |")
    out += ["", "### 単勝：予想レポートに載せる買い方の候補\n",
            "「ケリー基準（上限つき）」は、1 レースの予算に min(1, ケリー基準の割合 ÷ 5%) を掛けた額を買う方法です"
            "（自信が大きい馬ほど多く、最大で予算いっぱい）。\n",
            f"| 選び方 | 配り方 | 点数 | {HEAD} | 1割ずれ | 2割ずれ | 100%超の年 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("win_rules", []):
        out.append(f"| {r['rule']} | {r['weight']} | {r['bets']:,} | {_cells(r)} | {_pc(r['noise10'])} | "
                   f"{_pc(r['noise20'])} | {r['years_above']}/{r['years']} |")
    out += ["", "## 4. 予測モデルの組み立てを変える\n",
            "「予測のずれ」は小さいほど良い指標です（勝ち馬につけた確率の対数、レース平均）。\n",
            f"| モデル | 予測のずれ | 期待値1.00以上: {HEAD} | 期待値1.10以上: 回収率 | 偶然の幅 | {SPLIT}年〜 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("stage2", []):
        b = r["ev1.10"]
        tail = (f"**{_pc(b['all']['roi'])}** | {_pc(b['all']['lo'])}〜{_pc(b['all']['hi'])} | {_pc(b['conf']['roi'])}"
                if b["all"]["races"] >= 100 else "該当ほぼなし | — | —")
        out.append(f"| {r['variant']} | {r['logloss']:.4f} | {_cells(r['ev1.00'])} | {tail} |")
    out += ["", "## 5. 2着・3着の確率の出し方を変える（馬連・ワイド・三連複）\n",
            "「三連複の当てやすさ」は大きい（0 に近い）ほど良い指標です。\n",
            f"| 方法 | 三連複の当てやすさ | 期待値1.00以上: {HEAD} | 期待値1.15以上: 回収率 | {SPLIT}年〜 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("place_model", []):
        b = r["期待値1.15以上"]
        out.append(f"| {r['variant']} | {r['trio_loglik']:.3f} | {_cells(r['期待値1.00以上'])} | "
                   f"**{_pc(b['all']['roi'])}** | {_pc(b['conf']['roi'])} |")
    out.append("")
    text = "\n".join(out)
    (ROOT / "reports" / "experiments.md").write_text(text)
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return text


def load() -> tuple[pd.DataFrame, dict, bool]:
    o = pd.read_pickle(WORK / "oos.pkl")
    art = pipeline.load_artifacts()
    hp = WORK / "lab_heads.pkl"
    heads = hp.exists()
    if heads:
        o = o.merge(pd.read_pickle(hp), on=["race_id", "horse_id"], how="left")
        o = o[o["s_rank"].notna()]
    _, _, payouts = features.load_raw()
    o = summary._win_frame(o, payouts)
    o["year"] = o["date"].dt.year
    return o, {"art": art, "payouts": payouts}, heads


def main(argv: list[str] | None = None) -> int:
    only = set(argv or sys.argv[1:])
    o, ctx, heads = load()
    art, payouts = ctx["art"], ctx["payouts"]
    prev = json.loads((ROOT / "reports" / "experiments.json").read_text()) \
        if only and (ROOT / "reports" / "experiments.json").exists() else {}
    res = dict(prev)
    run = lambda k: not only or k in only
    tables = pipeline.odds_tables(features.load_odds())
    pay = strategy.payout_maps(payouts[payouts["race_id"].isin(tables)])
    if run("win"):
        res["win_staking"] = exp_win_staking(o)
        res["win_slices"] = exp_win_slices(o)
        res["combined_rule"] = exp_combined_rule(o, res["win_slices"])
        res["win_grid"] = exp_win_grid(o)
        res["win_rules"] = exp_win_rules(o)
    if run("slips"):
        ex = exotic.build(o, tables, pay, art["lam2"], art["lam3"])
        top = o.loc[o.groupby("race_id")["p_final"].idxmax()]
        res["slips"] = exp_slips(ex, top.set_index("race_id")["number"].astype(int).to_dict())
    if run("stage2"):
        res["stage2"] = exp_stage2(o, heads)
    if run("place"):
        res["place_model"] = exp_place_model(o, tables, pay, art["lam2"], art["lam3"], heads)
    res.setdefault("takeaways", [])
    write_report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
