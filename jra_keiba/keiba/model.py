"""Two-stage win-probability model.

Stage 1  LightGBM on pre-race fundamentals (no odds), two heads: win and top-3.
Stage 2  Conditional (multinomial) logit across the runners of a race that blends
         the stage-1 scores with the market's implied probability (Benter-style):
             P_i ∝ exp(a1*s_win_i + a2*s_top3_i + b*log(q_i))
         where q_i is the odds-implied probability. Without odds, b is dropped.

Stage 2 is always fitted on out-of-fold stage-1 scores from earlier years, so the
reported probabilities and backtests are out of sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.04, num_leaves=47, min_data_in_leaf=120,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    max_bin=127, verbose=-1, num_threads=0, seed=7,
)
# Residual head: starts from the market's implied probability (as init score) and only
# learns where the market is systematically off. Heavily regularised: the signal is tiny.
RES_PARAMS = dict(LGB_PARAMS, learning_rate=0.02, num_leaves=15, min_data_in_leaf=400,
                  feature_fraction=0.5, lambda_l2=30.0)
N_ROUNDS = {"win": 450, "top3": 550, "res": 150}
MARKET_FEATS = ["m", "mkt_rank", "mkt_gap", "mkt_top_p"]


def add_market_features(df: pd.DataFrame, odds_col: str = "win_odds") -> pd.DataFrame:
    """m = log odds-implied probability (normalised in the race) plus simple shape features.
    Rows of races with a missing price get NaN."""
    df = df.copy()
    inv = 1.0 / df[odds_col].astype(float)
    g = inv.groupby(df["race_id"])
    ok = g.transform(lambda s: s.notna().all())
    p = (inv / g.transform("sum")).where(ok)
    df["m"] = np.log(p)
    df["mkt_rank"] = p.groupby(df["race_id"]).rank(ascending=False, method="min")
    top = p.groupby(df["race_id"]).transform("max")
    df["mkt_gap"] = df["m"] - np.log(top)
    df["mkt_top_p"] = top
    return df


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train_stage1(df: pd.DataFrame, feats: list[str], rounds: dict[str, int] | None = None
                 ) -> dict[str, lgb.Booster]:
    rounds = rounds or N_ROUNDS
    models = {}
    for target in ("win", "top3"):
        ds = lgb.Dataset(df[feats], label=df[target], free_raw_data=False)
        models[target] = lgb.train(LGB_PARAMS, ds, num_boost_round=rounds[target])
    if "m" in df.columns:
        d = df[np.isfinite(df["m"])]
        if len(d) > 5000:
            ds = lgb.Dataset(d[feats + MARKET_FEATS], label=d["win"], init_score=_logit(np.exp(d["m"].to_numpy())))
            models["res"] = lgb.train(RES_PARAMS, ds, num_boost_round=rounds.get("res", N_ROUNDS["res"]))
    return models


def predict_stage1(models: dict[str, lgb.Booster], df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["s_win"] = _logit(models["win"].predict(df[feats]))
    out["s_top3"] = _logit(models["top3"].predict(df[feats]))
    out["s_res"] = 0.0
    if "res" in models and "m" in df.columns:
        ok = np.isfinite(df["m"].to_numpy(dtype=float))
        if ok.any():   # raw score excludes the init score: this is the correction to the market
            out.loc[ok, "s_res"] = models["res"].predict(df.loc[ok, feats + MARKET_FEATS], raw_score=True)
    return out


def tune_rounds(df: pd.DataFrame, feats: list[str], valid_year: int) -> dict[str, int]:
    """Pick the number of boosting rounds by early stopping on one held-out year."""
    tr, va = df[df["year"] < valid_year], df[df["year"] == valid_year]
    rounds = {}
    for target in ("win", "top3"):
        dtr = lgb.Dataset(tr[feats], label=tr[target])
        dva = lgb.Dataset(va[feats], label=va[target], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds[target] = int(max(150, m.best_iteration))
    if "m" in df.columns:
        trm, vam = tr[np.isfinite(tr["m"])], va[np.isfinite(va["m"])]
        if len(trm) > 5000 and len(vam) > 1000:
            cols = feats + MARKET_FEATS
            dtr = lgb.Dataset(trm[cols], label=trm["win"], init_score=_logit(np.exp(trm["m"].to_numpy())))
            dva = lgb.Dataset(vam[cols], label=vam["win"], init_score=_logit(np.exp(vam["m"].to_numpy())),
                              reference=dtr)
            m = lgb.train(RES_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds["res"] = int(min(600, max(30, m.best_iteration)))
    return rounds


def walk_forward(df: pd.DataFrame, feats: list[str], test_years: list[int],
                 rounds: dict[str, int] | None = None, log=print) -> pd.DataFrame:
    """Out-of-fold stage-1 scores: for each test year train on all earlier years."""
    parts = []
    for y in test_years:
        tr, te = df[df["year"] < y], df[df["year"] == y]
        if not len(te) or not len(tr):
            continue
        models = train_stage1(tr, feats, rounds)
        parts.append(predict_stage1(models, te, feats))
        log(f"  fold {y}: train={len(tr):,} test={len(te):,}")
    return pd.concat(parts) if parts else pd.DataFrame(columns=["s_win", "s_top3", "s_res"])


# ------------------------------------------------------------------ stage 2

def market_logprob(df: pd.DataFrame, odds_col: str = "win_odds") -> pd.Series:
    """log of odds-implied probability, normalised within the race (takeout removed)."""
    inv = 1.0 / df[odds_col].astype(float)
    tot = inv.groupby(df["race_id"]).transform("sum")
    return np.log(inv / tot)


def _race_index(race_ids: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(race_ids, sort=False)
    return codes, len(uniq)


def softmax_by_race(util: np.ndarray, codes: np.ndarray, n_races: int) -> np.ndarray:
    mx = np.full(n_races, -np.inf)
    np.maximum.at(mx, codes, util)
    e = np.exp(util - mx[codes])
    den = np.bincount(codes, weights=e, minlength=n_races)
    return e / den[codes]


@dataclass
class Stage2:
    """Conditional logit weights. ``cols`` name the per-runner inputs."""
    cols: list[str]
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, df: pd.DataFrame, l2: float = 1e-4) -> "Stage2":
        X = df[self.cols].to_numpy(dtype=float)
        codes, n = _race_index(df["race_id"])
        y = df["win"].to_numpy(dtype=float)
        n_win = np.bincount(codes, weights=y, minlength=n)
        ok = (n_win[codes] > 0)                      # races with a recorded winner
        X, y = X[ok], y[ok]
        codes, n = _race_index(df["race_id"][ok])
        w = y / np.bincount(codes, weights=y, minlength=n)[codes]   # dead heats share the win

        def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
            p = softmax_by_race(X @ theta, codes, n)
            loss = -(w * np.log(np.clip(p, 1e-12, None))).sum() / n + l2 * theta @ theta
            grad = -(X * (w - p)[:, None]).sum(axis=0) / n + 2 * l2 * theta
            return loss, grad

        x0 = np.full(len(self.cols), 0.5)
        res = minimize(nll, x0, jac=True, method="L-BFGS-B")
        self.coef = res.x
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        codes, n = _race_index(df["race_id"])
        return softmax_by_race(df[self.cols].to_numpy(dtype=float) @ self.coef, codes, n)


def race_logloss(df: pd.DataFrame, p: np.ndarray) -> float:
    """Mean negative log-likelihood of the winner per race."""
    w = df["win"].to_numpy(dtype=float)
    n = df["race_id"].nunique()
    return float(-(w * np.log(np.clip(p, 1e-12, None))).sum() / max(n, 1))


def fit_place_discount(df: pd.DataFrame, p_col: str = "p") -> tuple[float, float]:
    """Fit the discounted-Harville exponents for 2nd and 3rd place by maximum likelihood."""
    d = df[["race_id", p_col, "finish"]].dropna()
    codes, n = _race_index(d["race_id"])
    p = d[p_col].to_numpy(dtype=float)
    fin = d["finish"].to_numpy()

    def nll(lam: float, place: int) -> float:
        a = p ** lam
        ahead = fin < place                          # runners already placed
        den = np.bincount(codes, weights=np.where(ahead, 0.0, a), minlength=n)
        hit = fin == place
        return float(-np.log(np.clip(a[hit] / den[codes][hit], 1e-12, None)).mean())

    lam2 = minimize_scalar(lambda x: nll(x, 2), bounds=(0.3, 1.3), method="bounded").x
    lam3 = minimize_scalar(lambda x: nll(x, 3), bounds=(0.3, 1.3), method="bounded").x
    return float(lam2), float(lam3)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t.reset_index(drop=True)


def group_contributions(model: lgb.Booster, X: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Per-runner SHAP contributions of the win model, summed by feature group."""
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    cols = list(X.columns)
    out = pd.DataFrame(index=X.index)
    for g, names in groups.items():
        idx = [cols.index(c) for c in names if c in cols]
        out[g] = contrib[:, idx].sum(axis=1) if idx else 0.0
    return out
