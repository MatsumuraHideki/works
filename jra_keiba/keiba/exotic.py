"""Multi-horse bets on races that have a full odds table.

Win probabilities are turned into 馬連 / ワイド / 三連複 ... probabilities with a discounted
Harville formula (see betting.py); ``build`` lines them up with each ticket's odds and
payout so selection rules can be backtested, and ``evaluate`` reports returns by expected
value threshold.

The model's "expected value" for these tickets is not a true expected value: tickets with
model EV >= 1 have returned about 0.98 per unit, not the ~1.24 the model claims. It is
useful for ranking (returns rise with it), not as a promise.

``evaluate`` also reports a blended variant, p' ∝ p_model^a * q_market^b with (a, b)
fitted per bet type on the other half of the races. Because q ∝ 1/odds and a + b ≈ 1, its
EV is nearly a monotone function of the raw EV, so those rows are the high-EV tail of the
raw rule rather than an independent method; they rest on few races and are not used for
live predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import betting

ROOT = Path(__file__).resolve().parent.parent
CAL_PATH = ROOT / "reports" / "exotic_calibration.json"
TYPES = ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
CORE = ["馬連", "ワイド", "三連複"]          # pools collected for every open-class race


def blend(p: np.ndarray, q: np.ndarray, a: float, b: float) -> np.ndarray:
    """Blended ticket probabilities, keeping the total of the model's probabilities."""
    w = np.exp(a * np.log(np.clip(p, 1e-12, None)) + b * np.log(np.clip(q, 1e-12, None)))
    return w / w.sum() * p.sum()


def race_tables(probs: dict, odds: dict, winners: dict | None = None) -> dict[str, dict]:
    """Aligned arrays per bet type for one race: tickets, model p, market q, odds, hit mask."""
    out = {}
    for bt in TYPES:
        pt, ot = probs.get(bt, {}), odds.get(bt, {})
        keys = [k for k in pt if k in ot and ot[k] and ot[k] > 1.0]
        if len(keys) < 2:
            continue
        p = np.array([pt[k] for k in keys])
        o = np.array([ot[k] for k in keys], dtype=float)
        q = 1.0 / o
        q = q / q.sum() * p.sum()
        hit = np.array([k in winners.get(bt, {}) for k in keys]) if winners is not None else None
        ret = np.array([winners.get(bt, {}).get(k, 0.0) / 100.0 for k in keys]) if winners is not None else None
        out[bt] = {"keys": keys, "p": p, "q": q, "odds": o, "hit": hit, "ret": ret}
    return out


def fit_type(tabs: list[dict]) -> tuple[float, float]:
    """Maximum-likelihood (a, b) for one bet type over many races."""
    tabs = [t for t in tabs if t["hit"] is not None and t["hit"].any()]
    if len(tabs) < 30:
        return 0.0, 1.0
    lp = [np.log(np.clip(t["p"], 1e-12, None)) for t in tabs]
    lq = [np.log(np.clip(t["q"], 1e-12, None)) for t in tabs]
    hits = [t["hit"] for t in tabs]

    def nll(theta: np.ndarray) -> float:
        a, b = theta
        tot = 0.0
        for x, y, h in zip(lp, lq, hits):
            u = a * x + b * y
            u = u - u.max()
            tot -= (u[h] - np.log(np.exp(u).sum())).mean()
        return tot / len(tabs) + 1e-3 * (a * a + (b - 1.0) ** 2)

    res = minimize(nll, np.array([0.3, 0.7]), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    a, b = res.x
    return float(max(a, 0.0)), float(b)


def build(oos: pd.DataFrame, odds_tables: dict, pay: dict, lam2: float, lam3: float, p_col: str = "p_final"
          ) -> dict[str, dict]:
    """{race_id: {'date', 'grade', bet_type: table}} for races with a full odds table."""
    out = {}
    for rid, g in oos[oos["race_id"].isin(odds_tables)].groupby("race_id", sort=False):
        g = g.dropna(subset=["number"])
        if len(g) < 5:
            continue
        probs = betting.bet_probabilities(g[p_col].to_numpy(), [int(x) for x in g["number"]], lam2, lam3)
        winners = {bt: pay.get((rid, bt), {}) for bt in TYPES}
        tabs = race_tables(probs, odds_tables[rid], winners)
        tabs["date"], tabs["grade"] = g["date"].iloc[0], g["grade"].iloc[0]
        out[rid] = tabs
    return out


def fit_all(races: dict[str, dict], ids: list[str]) -> dict[str, list[float]]:
    return {bt: list(fit_type([races[r][bt] for r in ids if bt in races[r]])) for bt in TYPES}


def evaluate(races: dict[str, dict], thresholds=(0.9, 1.0, 1.05, 1.15)) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Two-fold out-of-sample test: calibrate on one half of the races (by date), bet the other.

    Returns (summary rows per bet type and EV threshold, calibration fitted on all races,
    per-race records). The "全買い目" rows are NOT an unselected baseline: they hold every
    ticket to which the model gives at least 0.5% probability (about a quarter of all
    tickets). The return of truly every ticket is in reports/summary.md.
    """
    ids = sorted(races, key=lambda r: races[r]["date"])
    half = len(ids) // 2
    folds = [(ids[:half], ids[half:]), (ids[half:], ids[:half])]
    rec = []
    for train, test in folds:
        cal = fit_all(races, train)
        for rid in test:
            for bt in TYPES:
                t = races[rid].get(bt)
                if t is None:
                    continue
                a, b = cal[bt]
                pc = blend(t["p"], t["q"], a, b)
                ev = pc * t["odds"]
                ev_raw = t["p"] * t["odds"]
                base = t["p"] >= 0.005
                rec.append((rid, bt, "全買い目", 0.0, int(base.sum()), float(t["ret"][base].sum()),
                            int((t["ret"][base] > 0).sum())))
                for thr in thresholds:
                    for name, e, pp in (("補正後", ev, pc), ("補正なし", ev_raw, t["p"])):
                        sel = (e >= thr) & (pp >= 0.005)
                        if sel.any():
                            rec.append((rid, bt, name, thr, int(sel.sum()), float(t["ret"][sel].sum()),
                                        int((t["ret"][sel] > 0).sum())))
    df = pd.DataFrame(rec, columns=["race_id", "bet_type", "kind", "thr", "n", "ret", "hits"])
    rows = []

    def row(kind, thr, bt, g):
        by_race = g.groupby("race_id").agg(n=("n", "sum"), r=("ret", "sum"))
        lo, hi = boot_ci(by_race["n"].to_numpy(dtype=float), by_race["r"].to_numpy())
        return {"方式": kind, "期待値の基準": "-" if kind == "全買い目" else f"{thr:.2f}以上", "券種": bt,
                "対象レース": g["race_id"].nunique(), "購入点数": int(g["n"].sum()),
                "的中率": g["hits"].sum() / g["n"].sum(), "回収率": g["ret"].sum() / g["n"].sum(),
                "95%下限": lo, "95%上限": hi}

    if len(df):
        for (kind, thr, bt), g in df.groupby(["kind", "thr", "bet_type"], sort=False):
            rows.append(row(kind, thr, bt, g))
        for (kind, thr), g in df.groupby(["kind", "thr"], sort=False):
            rows.append(row(kind, thr, "全券種", g))
            core = g[g["bet_type"].isin(CORE)]
            if len(core):
                rows.append(row(kind, thr, "馬連+ワイド+三連複", core))
    return pd.DataFrame(rows), fit_all(races, ids), df


def boot_ci(stake: np.ndarray, ret: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    n = len(stake)
    if n < 20 or stake.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    s, r = stake[idx].sum(axis=1), ret[idx].sum(axis=1)
    roi = r / np.where(s > 0, s, np.nan)
    return float(np.nanpercentile(roi, 2.5)), float(np.nanpercentile(roi, 97.5))


def save_calibration(cal: dict, n_races: int) -> None:
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps({"races": n_races, "weights": cal}, ensure_ascii=False, indent=1))


def load_calibration() -> dict | None:
    return json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else None


def calibrated_probs(probs: dict, odds: dict, cal: dict) -> dict:
    """Apply the stored (a, b) to one upcoming race. Types without odds are dropped."""
    out = {}
    for bt, t in race_tables(probs, odds).items():
        a, b = cal["weights"].get(bt, [0.0, 1.0])
        pc = blend(t["p"], t["q"], a, b)
        out[bt] = dict(zip(t["keys"], pc.tolist()))
    return out
