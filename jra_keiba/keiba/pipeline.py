"""Train the model, evaluate it out of sample and write the backtest report.

Usage:
    python -m keiba.pipeline [--oof-years 6] [--no-report]

Outputs:
    data/work/model.pkl      trained artefacts (not committed)
    reports/backtest.md      out-of-sample accuracy and betting backtest
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import betting, features, model

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
REPORTS = ROOT / "reports"
ORDERED = ("馬単", "三連単")


def fold_ids(dates: pd.Series, fold_months: int) -> pd.Series:
    """Walk-forward period of each row: calendar years, or shorter blocks when data is short."""
    return ((dates.dt.year * 12 + dates.dt.month - 1) // fold_months).astype(int)


def fold_label(fid: int, fold_months: int) -> str:
    y, m = divmod(fid * fold_months, 12)
    return str(y) if fold_months == 12 else f"{y}-{m + 1:02d}〜"


def fit_all(raw: Path = features.RAW, oof_years: int = 6, fold_months: int | None = None, log=print) -> dict:
    t0 = time.time()
    df = features.build_features(raw)
    feats = features.feature_columns()
    groups = {g: list(v) for g, v in features.FEATURE_GROUPS.items()}
    log(f"features: {len(df):,} runners, {df['race_id'].nunique():,} races, {len(feats)} features "
        f"({time.time() - t0:.0f}s)")
    hist = df[(~df["is_card"]) & df["win"].notna()].copy()
    span_years = hist["date"].dt.year.nunique()
    if fold_months is None:          # yearly folds need several seasons; fall back to shorter blocks
        fold_months = 12 if span_years >= 5 else (6 if span_years >= 2 else 2)
    hist["year"] = fold_ids(hist["date"], fold_months)   # "year" = walk-forward period id
    hist["m"] = model.market_logprob(hist)
    years = sorted(hist["year"].unique())
    log(f"walk-forward periods: {len(years)} x {fold_months} months")
    test_years = years[1:][-oof_years:]
    rounds = model.tune_rounds(hist, feats, years[-2]) if len(years) >= 3 else dict(model.N_ROUNDS)
    log(f"boosting rounds: {rounds}")
    oof = model.walk_forward(hist, feats, test_years, rounds, log=log)
    hist = hist.join(oof)
    o = hist.dropna(subset=["s_win", "m"]).copy()
    o = o[np.isfinite(o["m"])]

    # stage 2, walk-forward: weights for year Y come from out-of-fold scores of earlier years
    o["p_fund"] = np.nan
    o["p_final"] = np.nan
    for y in test_years[1:]:
        tr, te = o[o["year"] < y], o[o["year"] == y]
        if not len(tr) or not len(te):
            continue
        o.loc[te.index, "p_fund"] = model.Stage2(["s_win", "s_top3"]).fit(tr).predict(te)
        o.loc[te.index, "p_final"] = model.Stage2(["s_win", "s_top3", "m"]).fit(tr).predict(te)
    oos = o.dropna(subset=["p_final"]).copy()
    oos["p_market"] = np.exp(oos["m"])

    s2_fund = model.Stage2(["s_win", "s_top3"]).fit(o)
    s2_full = model.Stage2(["s_win", "s_top3", "m"]).fit(o)
    src = oos if len(oos) else o.assign(p_final=s2_full.predict(o))
    lam2, lam3 = model.fit_place_discount(src, "p_final")
    log(f"stage2 fund={np.round(s2_fund.coef, 3)} full={np.round(s2_full.coef, 3)} lam2={lam2:.3f} lam3={lam3:.3f}")

    models = model.train_stage1(hist, feats, rounds)
    art = {
        "models": {k: m.model_to_string() for k, m in models.items()},
        "feats": feats, "groups": groups, "rounds": rounds,
        "s2_fund": s2_fund.coef, "s2_full": s2_full.coef, "lam2": lam2, "lam3": lam3,
        "trained_through": str(hist["date"].max().date()), "n_races": int(hist["race_id"].nunique()),
        "years": [int(y) for y in years], "oos_years": [int(y) for y in test_years[1:]],
        "fold_months": fold_months, "data_from": str(hist["date"].min().date()),
    }
    log(f"fit_all done in {time.time() - t0:.0f}s")
    return {"art": art, "oos": oos}


def save_artifacts(art: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "model.pkl", "wb") as fh:
        pickle.dump(art, fh)


def load_artifacts() -> dict | None:
    p = WORK / "model.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def boosters(art: dict) -> dict[str, lgb.Booster]:
    return {k: lgb.Booster(model_str=s) for k, s in art["models"].items()}


# ------------------------------------------------------------------ backtests

def win_bet_table(oos: pd.DataFrame, p_col: str = "p_final") -> pd.DataFrame:
    """Flat-stake 単勝 backtest at final odds for several EV thresholds."""
    ev = oos[p_col] * oos["win_odds"]
    rows = []
    for thr in (0.0, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
        sel = oos[ev.notna()] if thr == 0 else oos[(ev >= thr) & (oos[p_col] >= 0.01)]
        if not len(sel):
            continue
        ret = (sel["win"] * sel["win_odds"]).sum()
        rows.append({"EV閾値": "全頭" if thr == 0 else f"{thr:.2f}以上", "購入点数": len(sel),
                     "的中率": sel["win"].mean(), "回収率": ret / len(sel),
                     "平均オッズ": sel["win_odds"].mean()})
    return pd.DataFrame(rows)


def parse_combo(bet_type: str, combo: str) -> tuple[int, ...]:
    nums = tuple(int(x) for x in str(combo).replace("→", "-").split("-") if x.strip().isdigit())
    return nums if bet_type in ORDERED or bet_type in ("単勝", "複勝") else tuple(sorted(nums))


def odds_tables(odds_df: pd.DataFrame, use_max: bool = False) -> dict[str, dict]:
    """{race_id: {bet_type: {combo: decimal odds}}}; range bets use the low end."""
    out: dict[str, dict] = {}
    col = "odds_max" if use_max else "odds"
    for (rid, bt), g in odds_df.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {parse_combo(bt, c): float(o) for c, o in zip(g["combo"], g[col])}
    return out


def exotic_backtest(oos: pd.DataFrame, odds_df: pd.DataFrame, payouts: pd.DataFrame, lam2: float, lam3: float,
                    min_ev: float = 1.15, p_col: str = "p_final") -> tuple[pd.DataFrame, pd.DataFrame]:
    """All bet types on races that have a full final-odds table (graded races).

    Tickets are chosen with the same rule as live predictions and settled with the
    official payouts. Returns (per-bet-type summary, per-ticket log).
    """
    tables = odds_tables(odds_df)
    pay = payouts[payouts["race_id"].isin(tables)].copy()
    pay["key"] = [parse_combo(b, c) for b, c in zip(pay["bet_type"], pay["combo"])]
    pay_map = {(r, b, k): v / 100.0 for r, b, k, v in zip(pay["race_id"], pay["bet_type"], pay["key"], pay["payout"])}
    log = []
    for rid, g in oos[oos["race_id"].isin(tables)].groupby("race_id", sort=False):
        g = g.dropna(subset=["number"])
        if len(g) < 5:
            continue
        probs = betting.bet_probabilities(g[p_col].to_numpy(), [int(x) for x in g["number"]], lam2, lam3)
        cand = betting.evaluate_bets(probs, tables[rid])
        picked = betting.select_bets(cand, min_ev=min_ev)
        picked = betting.size_bets(picked)
        for b in picked:
            ret = pay_map.get((rid, b.bet_type, b.combo), 0.0)
            log.append({"race_id": rid, "date": g["date"].iloc[0], "grade": g["grade"].iloc[0],
                        "bet_type": b.bet_type, "combo": b.label(), "prob": b.prob, "odds": b.odds,
                        "ev": b.ev, "stake": b.stake, "hit": float(ret > 0), "ret_flat": ret,
                        "ret_kelly": ret * b.stake})
    log_df = pd.DataFrame(log)
    if not len(log_df):
        return pd.DataFrame(), log_df
    rows = []
    for bt, g in list(log_df.groupby("bet_type", sort=False)) + [("合計", log_df)]:
        rows.append({"券種": bt, "購入点数": len(g), "的中率": g["hit"].mean(),
                     "回収率(均等買い)": g["ret_flat"].mean(),
                     "回収率(ケリー配分)": g["ret_kelly"].sum() / max(g["stake"].sum(), 1),
                     "平均オッズ": g["odds"].mean()})
    return pd.DataFrame(rows), log_df


def _md(df: pd.DataFrame, pct: tuple[str, ...] = ()) -> str:
    if not len(df):
        return "(該当なし)\n"
    d = df.copy()
    for c in d.columns:
        if c in pct or "率" in c:
            d[c] = [f"{v * 100:.1f}%" if pd.notna(v) else "" for v in d[c]]
        elif d[c].dtype.kind == "f":
            d[c] = [f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}" for v in d[c]]
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_report(res: dict, raw: Path = features.RAW, log=print) -> str:
    art, oos = res["art"], res["oos"]
    out = ["# バックテスト結果（学習に使っていない期間での検証）\n"]
    fm = art["fold_months"]
    labels = [fold_label(y, fm) for y in art["oos_years"]]
    out.append(f"- 学習データ: {art['data_from']}〜{art['trained_through']} / {art['n_races']:,} レース（平地のみ、新馬〜G1）")
    out.append(f"- 検証期間: {'、'.join(labels)}（{fm}か月ごとに区切り、各期間をそれより前のデータだけで学習したモデルで予測）")
    out.append(f"- 特徴量 {len(art['feats'])} 個 / 着順モデルの補正係数 λ2={art['lam2']:.2f}, λ3={art['lam3']:.2f}\n")
    if not len(oos):
        out.append("検証に必要な年数のデータがまだありません。\n")
        text = "\n".join(out)
        return text
    graded = oos[oos["grade"].isin(["G1", "G2", "G3"])]

    out.append("## 1. 予測精度（勝ち馬の対数損失。小さいほど良い）\n")
    rows = []
    for label, d in [("全レース", oos), ("重賞(G1-G3)", graded)] + [(fold_label(y, fm), oos[oos["year"] == y]) for y in art["oos_years"]]:
        if not len(d):
            continue
        top = d.loc[d.groupby("race_id")["p_final"].idxmax()]
        fav = d.loc[d.groupby("race_id")["p_market"].idxmax()]
        rows.append({"対象": label, "レース数": d["race_id"].nunique(),
                     "市場オッズのみ": model.race_logloss(d, d["p_market"].to_numpy()),
                     "モデル単独": model.race_logloss(d, d["p_fund"].to_numpy()),
                     "モデル+オッズ": model.race_logloss(d, d["p_final"].to_numpy()),
                     "本命的中率(モデル+オッズ)": top["win"].mean(), "1番人気的中率": fav["win"].mean()})
    out.append(_md(pd.DataFrame(rows)))
    out.append("「モデル+オッズ」が「市場オッズのみ」より小さければ、モデルがオッズに含まれない情報を足せています。\n")

    out.append("## 2. 確率の当たり具合（モデル+オッズ、全レース）\n")
    cal = model.calibration_table(oos["p_final"].to_numpy(), oos["win"].to_numpy())
    cal.columns = ["頭数", "予測勝率", "実際の勝率"]
    out.append(_md(cal, pct=("予測勝率", "実際の勝率")))

    out.append("## 3. 単勝を期待値で選んだ場合（確定オッズ、100円均等買い）\n")
    out.append("### 全レース\n")
    out.append(_md(win_bet_table(oos)))
    out.append("### 重賞(G1-G3)のみ\n")
    out.append(_md(win_bet_table(graded)))
    out.append("### モデル単独の確率で選んだ場合（全レース、参考）\n")
    out.append(_md(win_bet_table(oos, "p_fund")))

    odds_df = features.load_odds(raw)
    if len(odds_df):
        _, _, payouts = features.load_raw(raw)
        out.append("## 4. 全券種を期待値で選んだ場合（重賞、確定オッズ全通り）\n")
        for thr in (1.05, 1.15, 1.3):
            summ, _ = exotic_backtest(oos, odds_df, payouts, art["lam2"], art["lam3"], min_ev=thr)
            n_races = oos[oos["race_id"].isin(set(odds_df["race_id"]))]["race_id"].nunique()
            out.append(f"### 期待値 {thr:.2f} 以上の買い目（対象 {n_races} レース）\n")
            out.append(_md(summ))
    out.append("## 注意\n")
    out.append("- 検証は確定オッズで行っています。実際に買う時点のオッズは確定オッズと異なり、"
               "直前に人気が動くと期待値は変わります。")
    out.append("- JRA の控除率は 20〜27.5% あり、回収率 100% 超えは少数の買い目の当たり外れに大きく左右されます。"
               "点数が少ない行の回収率は偶然の幅が大きい数字です。")
    out.append("- 特徴量はすべてレース前日までの情報だけで作っています（当日の馬体重を除く）。"
               "未来の結果が混ざらないことは tests/test_features.py で確認しています。\n")
    text = "\n".join(out)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "backtest.md").write_text(text)
    log(f"wrote {REPORTS / 'backtest.md'}")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof-years", type=int, default=6)
    ap.add_argument("--fold-months", type=int, default=None, help="walk-forward block length (default: auto)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    res = fit_all(oof_years=args.oof_years, fold_months=args.fold_months)
    save_artifacts(res["art"])
    if not args.no_report:
        print(write_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
