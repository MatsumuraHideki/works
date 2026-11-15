"""Experimental stage-1 heads, out of fold (walk-forward by year).

    python -m keiba.lab_heads            -> data/work/lab_heads.pkl

Each head is trained for test year Y on races before Y only, like the production heads.
Heads:
  s_rank   LambdaRank on finishing position (1st=3, 2nd=2, 3rd=1), grouped by race
  s_winw   the win head with recent seasons weighted more (half-life 4 seasons)
  s_resw   the market-residual head with the same recency weights
  s_res3   a market-residual head for finishing in the first three
The number of boosting rounds per fold is the one the production fit chose for that fold
(data/work/fold_rounds.log), so nothing here is tuned on the test years.
"""
from __future__ import annotations

import ast
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features, model
from .pipeline import WORK

HALF_LIFE = 4.0
RANK_PARAMS = dict(model.LGB_PARAMS, objective="lambdarank", lambdarank_truncation_level=6,
                   label_gain=[0, 1, 3, 7])


def fold_rounds() -> dict[int, dict[str, int]]:
    out = {}
    for line in (WORK / "fold_rounds.log").read_text().splitlines():
        m = re.search(r"fold (\d+):.*rounds=(\{.*\})", line)
        if m:
            out[int(m.group(1))] = ast.literal_eval(m.group(2))
    return out


def main() -> int:
    t0 = time.time()
    df = features.build_features()
    feats = features.feature_columns()
    hist = df[(~df["is_card"]) & df["win"].notna()].copy()
    hist["year"] = hist["date"].dt.year
    hist = model.add_market_features(hist)
    hist = hist.sort_values(["date", "race_id"], kind="stable")
    rounds = fold_rounds()
    parts = []
    for y in sorted(rounds):
        if y < 2015:
            continue
        tr, te = hist[hist["year"] < y], hist[hist["year"] == y]
        r = rounds[y]
        w = 0.5 ** ((y - 1 - tr["year"].to_numpy()) / HALF_LIFE)
        out = te[["race_id", "horse_id"]].copy()
        # LambdaRank
        rel = np.select([tr["finish"] == 1, tr["finish"] == 2, tr["finish"] == 3], [3, 2, 1], 0)
        grp = tr.groupby("race_id", sort=False).size().to_numpy()
        ds = lgb.Dataset(tr[feats], label=rel, group=grp)
        out["s_rank"] = lgb.train(RANK_PARAMS, ds, num_boost_round=r["win"]).predict(te[feats])
        # recency-weighted win head
        ds = lgb.Dataset(tr[feats], label=tr["win"], weight=w)
        out["s_winw"] = model._logit(lgb.train(model.LGB_PARAMS, ds, num_boost_round=r["win"]).predict(te[feats]))
        # market-residual heads
        ok = np.isfinite(tr["m"].to_numpy(dtype=float))
        trm, wm = tr[ok], w[ok]
        cols = feats + model.MARKET_FEATS
        tek = np.isfinite(te["m"].to_numpy(dtype=float))
        init = model._logit(np.exp(trm["m"].to_numpy()))
        ds = lgb.Dataset(trm[cols], label=trm["win"], init_score=init, weight=wm)
        out["s_resw"] = 0.0
        out.loc[tek, "s_resw"] = lgb.train(model.RES_PARAMS, ds, num_boost_round=r["res"]).predict(
            te.loc[tek, cols], raw_score=True)
        # top-3 residual: start from a crude market top-3 probability (3 x win share, capped)
        init3 = model._logit(np.clip(3.0 * np.exp(trm["m"].to_numpy()), 1e-4, 0.92))
        ds = lgb.Dataset(trm[cols], label=trm["top3"], init_score=init3)
        out["s_res3"] = 0.0
        out.loc[tek, "s_res3"] = lgb.train(model.RES_PARAMS, ds, num_boost_round=r["res"]).predict(
            te.loc[tek, cols], raw_score=True)
        parts.append(out)
        print(f"fold {y}: train={len(tr):,} test={len(te):,} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(parts).to_pickle(WORK / "lab_heads.pkl")
    print("HEADS DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
