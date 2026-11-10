"""Load scraped data and build leak-free pre-race features.

Every feature of a runner uses only information available before the race:
earlier races of the horse / jockey / trainer / sire, the race conditions and
the draw. Upcoming race cards are appended to the history and go through the
same code path, so training and prediction features are computed identically.

Feature groups (the "aspects" reported with each prediction):
    ability    Elo ratings, speed figures, earnings, class reached
    form       recent finishes, margins, layoff
    aptitude   record on this surface / distance / course / going
    people     jockey and trainer ratings and strike rates
    pedigree   sire and broodmare-sire progeny records
    race       pace set-up, draw, weights, field
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ID_COLS = {"race_id": str, "horse_id": str, "jockey_id": str, "trainer_id": str, "owner_id": str,
           "combo": str, "finish_raw": str}
GRADE_LEVEL = {"L": 5.5, "G3": 6.0, "G2": 7.0, "G1": 8.0, "重賞": 6.0}
CLASS_LEVEL = {"新馬": 1.0, "未勝利": 1.0, "1勝": 2.0, "2勝": 3.0, "3勝": 4.0, "OP": 5.0, "不明": 3.0}
GOING_CODE = {"良": 0, "稍重": 1, "重": 2, "不良": 3}


# ------------------------------------------------------------------ loading

def _read_all(kind: str, raw: Path = RAW) -> pd.DataFrame:
    files = sorted((raw / kind).glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype=ID_COLS) for f in files], ignore_index=True)


def load_raw(raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races = _read_all("races", raw).drop_duplicates("race_id")
    results = _read_all("results", raw).drop_duplicates(["race_id", "horse_id"])
    payouts = _read_all("payouts", raw)
    races["date"] = pd.to_datetime(races["date"])
    return races, results, payouts


def load_pedigree(raw: Path = RAW) -> pd.DataFrame:
    p = raw / "pedigree.csv.gz"
    if not p.exists():
        return pd.DataFrame(columns=["horse_id", "sire", "dam", "bms"])
    return pd.read_csv(p, dtype=str).drop_duplicates("horse_id")


def load_odds(raw: Path = RAW) -> pd.DataFrame:
    return _read_all("odds", raw)


def class_level(races: pd.DataFrame) -> pd.Series:
    lvl = races["class"].map(CLASS_LEVEL).fillna(3.0)
    g = races["grade"].map(GRADE_LEVEL)
    return g.fillna(lvl)


def build_frame(races: pd.DataFrame, results: pd.DataFrame, pedigree: pd.DataFrame | None = None,
                card_races: pd.DataFrame | None = None, card_runners: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """One row per runner, chronologically sorted, flat races only.

    ``card_*`` are upcoming races (no results yet); they are flagged ``is_card``.
    """
    races = races.copy()
    results = results.copy()
    races["is_card"] = False
    if card_races is not None and len(card_races):
        cr = card_races.copy()
        cr["date"] = pd.to_datetime(cr["date"])
        cr["is_card"] = True
        races = pd.concat([races[~races["race_id"].isin(cr["race_id"])], cr], ignore_index=True)
        results = pd.concat([results[~results["race_id"].isin(cr["race_id"])], card_runners],
                            ignore_index=True)
    races["level"] = class_level(races)
    keep = ["race_id", "date", "place", "surface", "direction", "inout", "distance", "going", "weather",
            "class", "grade", "level", "race_no", "name", "is_card", "cond"]
    df = results.merge(races[[c for c in keep if c in races.columns]], on="race_id", how="inner")
    df = df[df["surface"].isin(["芝", "ダ"])].copy()

    raw_fin = df["finish_raw"].fillna("").astype(str) if "finish_raw" in df else pd.Series("", index=df.index)
    scratched = raw_fin.str.contains("取|除") & ~df["is_card"]
    df = df[~scratched].copy()
    df["ran"] = (~df["is_card"]).astype(float)
    # field size = horses that actually started
    df["n_runners"] = df.groupby("race_id")["horse_id"].transform("size").astype(float)
    # did-not-finish / disqualified count as last
    fin = df["finish"].astype(float)
    fin = fin.where(fin.notna() | df["is_card"], df["n_runners"])
    df["finish"] = fin
    df["pos"] = (df["finish"] - 1) / (df["n_runners"] - 1).clip(lower=1)
    df["win"] = (df["finish"] == 1).astype(float).where(~df["is_card"])
    df["top2"] = (df["finish"] <= 2).astype(float).where(~df["is_card"])
    df["top3"] = (df["finish"] <= 3).astype(float).where(~df["is_card"])

    if pedigree is not None and len(pedigree):
        df = df.merge(pedigree[["horse_id", "sire", "bms"]], on="horse_id", how="left")
    else:
        df["sire"] = np.nan
        df["bms"] = np.nan
    df["number"] = df["number"].astype(float)
    df = df.sort_values(["date", "race_id", "number", "horse_id"], kind="stable").reset_index(drop=True)
    return df


# ------------------------------------------------------------------ speed figures

def add_speed_figures(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based figures: seconds per km faster than the course par, corrected by a
    per-day track variant.

    Pars and class effects for a season come from *earlier* seasons only (the first
    season in the data uses itself), so a figure never depends on later results.
    These are outcome columns: features only ever read them through past races.
    """
    key = ["place", "surface", "distance"]
    df = df.copy()
    df["_year"] = df["date"].dt.year
    ok = df["time_sec"].notna() & (df["ran"] == 1)
    km = df["distance"] / 1000.0
    years = sorted(df["_year"].unique())
    par_time = pd.Series(np.nan, index=df.index)
    par_l3f = pd.Series(np.nan, index=df.index)
    cls_eff = pd.Series(np.nan, index=df.index)
    resid = pd.Series(np.nan, index=df.index)
    for y in years:
        cur = df["_year"] == y
        base = (df["_year"] < y) if y != years[0] else cur
        g = df[base & ok].groupby(key)["time_sec"].agg(["median", "size"])
        g = g[g["size"] >= 30]["median"]
        idx = pd.MultiIndex.from_frame(df.loc[cur, key])
        par_time[cur] = g.reindex(idx).to_numpy()
        g3 = df[base & ok & df["last3f"].notna()].groupby(key)["last3f"].median()
        par_l3f[cur] = g3.reindex(idx).to_numpy()
        resid[cur] = (df.loc[cur, "time_sec"] - par_time[cur]) / km[cur]   # sec/km slower than par
        # class effect, so the day variant is not biased by the class mix of the card
        top = base & (df["finish"] <= 5) & resid.notna() & (df["ran"] == 1)
        if top.any():
            t = df.loc[top, ["race_id", "surface", "level"]].assign(r=resid[top])
            race_r = t.groupby("race_id").agg(r=("r", "mean"), surface=("surface", "first"), level=("level", "first"))
            ce = race_r.groupby(["surface", "level"])["r"].mean()
            cls_eff[cur] = ce.reindex(pd.MultiIndex.from_frame(df.loc[cur, ["surface", "level"]])).to_numpy()
    df["_resid"] = resid
    top = (df["finish"] <= 5) & df["_resid"].notna() & (df["ran"] == 1)
    t = df.loc[top, ["race_id", "date", "place", "surface"]].assign(dev=df.loc[top, "_resid"] - cls_eff[top].fillna(0.0))
    race_dev = t.groupby("race_id").agg(dev=("dev", "mean"), date=("date", "first"), place=("place", "first"),
                                        surface=("surface", "first"))
    variant = race_dev.groupby(["date", "place", "surface"])["dev"].median().rename("variant").reset_index()
    df = df.merge(variant, on=["date", "place", "surface"], how="left")
    df["fig"] = -(df["_resid"] - df["variant"].fillna(0.0))
    df.loc[df["fig"].abs() > 15, "fig"] = np.nan                    # eased / broken-down runs
    df["fig_l3f"] = -(df["last3f"] - par_l3f.to_numpy())
    df.loc[df["fig_l3f"].abs() > 8, "fig_l3f"] = np.nan
    # early position: first call as a fraction of the field (0 = led)
    first = pd.to_numeric(df["passing"].astype(str).str.split("-").str[0], errors="coerce")
    df["early_pos"] = ((first - 1) / (df["n_runners"] - 1).clip(lower=1)).clip(0, 1)
    return df.drop(columns=["_resid", "_year"])


# ------------------------------------------------------------------ Elo ratings

def add_elo(df: pd.DataFrame, k_horse: float = 28.0, k_other: float = 6.0) -> pd.DataFrame:
    """Multi-runner Elo for horses (overall and per surface), jockeys, trainers and sires.

    Ratings stored on a row are the values at the start of that race *day*: updates
    are applied once the whole day is over, so same-day results never feed a rating.
    """
    n = len(df)
    out = {name: np.full(n, np.nan) for name in ("elo_h", "elo_hs", "elo_j", "elo_t", "elo_sire", "elo_bms")}
    tables: dict[str, dict] = {name: {} for name in out}
    horse = df["horse_id"].to_numpy()
    surf = df["surface"].to_numpy()
    jockey = df["jockey_id"].to_numpy()
    trainer = df["trainer_id"].to_numpy()
    sire = df["sire"].to_numpy()
    bms = df["bms"].to_numpy()
    finish = df["finish"].to_numpy(dtype=float)
    is_card = df["is_card"].to_numpy()
    level = df["level"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    race_codes, starts = np.unique(df["race_id"].to_numpy(), return_index=True)
    order = np.argsort(starts)
    bounds = list(starts[order]) + [n]
    pending: list[tuple[str, object, float]] = []
    day = None
    for b in range(len(order)):
        lo, hi = bounds[b], bounds[b + 1]
        if dates[lo] != day:
            for name, key, d in pending:
                tab = tables[name]
                tab[key] = tab.get(key, 0.0) + d
            pending, day = [], dates[lo]
        idx = np.arange(lo, hi)
        keys = {
            "elo_h": [horse[i] for i in idx],
            "elo_hs": [(horse[i], surf[i]) for i in idx],
            "elo_j": [jockey[i] for i in idx],
            "elo_t": [trainer[i] for i in idx],
            "elo_sire": [(sire[i], surf[i]) if isinstance(sire[i], str) else None for i in idx],
            "elo_bms": [(bms[i], surf[i]) if isinstance(bms[i], str) else None for i in idx],
        }
        # a newcomer starts at the typical level of the class it is entered in
        base = 1500.0 + 60.0 * (level[lo] - 1.0)
        cur = {}
        for name, ks in keys.items():
            tab = tables[name]
            default = base if name in ("elo_h", "elo_hs") else 1500.0
            vals = np.array([tab.get(k, default) if k is not None else np.nan for k in ks])
            cur[name] = vals
            out[name][lo:hi] = vals
        if is_card[lo] or hi - lo < 2:
            continue
        f = finish[lo:hi]
        s = (f[None, :] > f[:, None]).astype(float) + 0.5 * (f[None, :] == f[:, None])
        np.fill_diagonal(s, 0.0)
        m = hi - lo
        for name, ks in keys.items():
            r = cur[name]
            valid = ~np.isnan(r)
            if valid.sum() < 2:
                continue
            rr = np.where(valid, r, 0.0)
            e = 1.0 / (1.0 + 10.0 ** ((rr[None, :] - rr[:, None]) / 400.0))
            w = (valid[None, :] & valid[:, None]).astype(float)
            np.fill_diagonal(w, 0.0)
            delta = ((s - e) * w).sum(axis=1) / max(valid.sum() - 1, 1)
            k = k_horse if name in ("elo_h", "elo_hs") else k_other
            # more weight to results in bigger fields
            k = k * np.sqrt(m / 12.0) * 2.0
            tab = tables[name]
            for j, key in enumerate(ks):
                if key is not None and valid[j]:
                    if key not in tab:
                        tab[key] = r[j]          # materialise the starting rating
                    pending.append((name, key, k * delta[j]))
    for name, arr in out.items():
        df[name] = arr
    return df


# ------------------------------------------------------------------ history helpers

def _prior_sum(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """Sum of ``col`` over earlier rows of the same group (current row excluded)."""
    v = df[col].fillna(0.0)
    return v.groupby([df[k] for k in keys], sort=False).cumsum() - v


def _rate(num: pd.Series, den: pd.Series, prior: float, k: float) -> pd.Series:
    """Shrunk rate: (num + k*prior) / (den + k)."""
    return (num + k * prior) / (den + k)


def _by_date_prior(df: pd.DataFrame, keys: list[str], cols: list[str], window: str | None = None
                   ) -> pd.DataFrame:
    """Per-entity sums over strictly earlier *dates* (so same-day races never leak).

    Returns a frame aligned with df, columns = cols.
    """
    tmp = df[keys + ["date"] + cols].copy()
    tmp[cols] = tmp[cols].fillna(0.0)
    daily = tmp.groupby(keys + ["date"], sort=True, dropna=False)[cols].sum().reset_index()
    if window is None:
        g = daily.groupby(keys, sort=False, dropna=False)[cols]
        prior = g.cumsum() - daily[cols]
    else:
        daily = daily.sort_values(keys + ["date"])
        parts = (daily.set_index("date").groupby(keys, sort=False, dropna=False)[cols]
                 .rolling(window, closed="left").sum())
        prior = parts.reset_index(drop=True).fillna(0.0)
        daily = daily.reset_index(drop=True)
    prior = pd.concat([daily[keys + ["date"]], prior.set_axis(cols, axis=1)], axis=1)
    merged = df[keys + ["date"]].merge(prior, on=keys + ["date"], how="left")
    merged.index = df.index
    return merged[cols]


def dist_band(d: pd.Series) -> pd.Series:
    return pd.cut(d, [0, 1300, 1650, 2050, 2500, 9999], labels=False).astype(float)


# ------------------------------------------------------------------ main feature builder

FEATURE_GROUPS: dict[str, list[str]] = {}


def _reg(group: str, names: list[str]) -> None:
    FEATURE_GROUPS.setdefault(group, [])
    for n in names:
        if n not in FEATURE_GROUPS[group]:
            FEATURE_GROUPS[group].append(n)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_speed_figures(df)
    df = add_elo(df)
    FEATURE_GROUPS.clear()
    df["band"] = dist_band(df["distance"])
    df["going_code"] = df["going"].map(GOING_CODE)
    df["wet"] = (df["going_code"] >= 2).astype(float).where(df["going_code"].notna())
    gh = df.groupby("horse_id", sort=False)

    # ---------- horse history (shift-based, fully vectorised)
    df["h_starts"] = _prior_sum(df, ["horse_id"], "ran")
    for c in ("win", "top3"):
        df[f"h_{c}_sum"] = _prior_sum(df, ["horse_id"], c)
    df["h_win_rate"] = _rate(df["h_win_sum"], df["h_starts"], 0.08, 3.0)
    df["h_top3_rate"] = _rate(df["h_top3_sum"], df["h_starts"], 0.24, 3.0)
    df["h_prize_sum"] = _prior_sum(df, ["horse_id"], "prize")
    df["h_prize_per_start"] = df["h_prize_sum"] / df["h_starts"].clip(lower=1)
    df["h_log_prize"] = np.log1p(df["h_prize_sum"])

    lag_cols = ["pos", "fig", "fig_l3f", "early_pos", "level", "finish", "n_runners", "distance",
                "weight_carried", "body_weight", "popularity", "win_odds"]
    lags = {c: [gh[c].shift(k) for k in range(1, 6)] for c in lag_cols}
    prev_date = gh["date"].shift(1)
    prev_surface = gh["surface"].shift(1)
    prev_jockey = gh["jockey_id"].shift(1)

    def stack(c: str, k: int) -> np.ndarray:
        return np.column_stack([s.to_numpy(dtype=float) for s in lags[c][:k]])

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows (debutants)
        df["pos_l1"] = lags["pos"][0]
        df["pos_l2"] = lags["pos"][1]
        df["pos_mean3"] = np.nanmean(stack("pos", 3), axis=1)
        df["pos_mean5"] = np.nanmean(stack("pos", 5), axis=1)
        df["pos_best5"] = np.nanmin(stack("pos", 5), axis=1)
        df["fig_l1"] = lags["fig"][0]
        df["fig_mean3"] = np.nanmean(stack("fig", 3), axis=1)
        df["fig_max3"] = np.nanmax(stack("fig", 3), axis=1)
        df["fig_max5"] = np.nanmax(stack("fig", 5), axis=1)
        df["fig_std5"] = np.nanstd(stack("fig", 5), axis=1)
        df["l3f_l1"] = lags["fig_l3f"][0]
        df["l3f_mean3"] = np.nanmean(stack("fig_l3f", 3), axis=1)
        df["l3f_max5"] = np.nanmax(stack("fig_l3f", 5), axis=1)
        df["style"] = np.nanmean(stack("early_pos", 5), axis=1)
        df["level_l1"] = lags["level"][0]
        df["level_max5"] = np.nanmax(stack("level", 5), axis=1)
        df["level_mean3"] = np.nanmean(stack("level", 3), axis=1)
        # class-adjusted form: finishing position weighted by the class it was achieved in
        lv, ps = stack("level", 5), stack("pos", 5)
        df["form_class5"] = np.nanmean((1 - ps) * lv, axis=1)
        df["form_class_best5"] = np.nanmax((1 - ps) * lv, axis=1)
        df["pop_l1"] = lags["popularity"][0]
        df["logodds_mean3"] = np.nanmean(np.log(stack("win_odds", 3)), axis=1)
    df["level_diff"] = df["level"] - df["level_l1"]
    df["days_since"] = (df["date"] - prev_date).dt.days.astype(float)
    df["log_days_since"] = np.log1p(df["days_since"])
    df["dist_diff"] = df["distance"] - lags["distance"][0]
    df["surface_switch"] = (prev_surface != df["surface"]).astype(float).where(prev_surface.notna())
    df["wc_diff"] = df["weight_carried"] - lags["weight_carried"][0]
    df["bw_l1"] = lags["body_weight"][0]
    df["bw_now"] = df["body_weight"].fillna(df["bw_l1"])         # unknown until ~1h before post
    df["wc_ratio"] = df["weight_carried"] / df["bw_now"]
    df["same_jockey"] = (prev_jockey == df["jockey_id"]).astype(float).where(prev_jockey.notna())
    df["is_female"] = (df["sex"] == "牝").astype(float)
    df["is_gelding"] = (df["sex"] == "セ").astype(float)
    df["month"] = df["date"].dt.month.astype(float)
    # age in a horse's career terms: starts per year of racing age
    _reg("ability", ["elo_h", "elo_hs", "fig_l1", "fig_mean3", "fig_max3", "fig_max5", "fig_std5",
                     "l3f_l1", "l3f_mean3", "l3f_max5", "h_win_rate", "h_top3_rate", "h_prize_per_start",
                     "h_log_prize", "level_max5", "level_mean3", "form_class5", "form_class_best5"])
    _reg("form", ["pos_l1", "pos_l2", "pos_mean3", "pos_mean5", "pos_best5", "days_since", "log_days_since",
                  "h_starts", "level_l1", "level_diff", "pop_l1", "logodds_mean3", "age"])

    # ---------- aptitude: record under today's conditions
    for name, keys in {
        "surf": ["horse_id", "surface"],
        "band": ["horse_id", "surface", "band"],
        "place": ["horse_id", "place", "surface"],
        "dir": ["horse_id", "surface", "direction"],
        "wet": ["horse_id", "surface", "wet"],
    }.items():
        st = _prior_sum(df, keys, "ran")
        t3 = _prior_sum(df, keys, "top3")
        df[f"apt_{name}_n"] = st
        df[f"apt_{name}_top3"] = _rate(t3, st, 0.24, 2.0)
        if name in ("surf", "band"):  # best earlier figure under the condition
            grp = [df[k] for k in keys]
            shifted = df["fig"].groupby(grp, sort=False).shift(1)
            df[f"apt_{name}_figmax"] = shifted.groupby(grp, sort=False).cummax()
        _reg("aptitude", [f"apt_{name}_n", f"apt_{name}_top3"])
    _reg("aptitude", ["apt_surf_figmax", "apt_band_figmax", "dist_diff", "surface_switch"])

    # ---------- people: jockey / trainer strike rates on earlier dates only
    for who, col in (("j", "jockey_id"), ("t", "trainer_id")):
        p = _by_date_prior(df, [col], ["ran", "win", "top3"], window="365D")
        df[f"{who}_n365"] = p["ran"]
        df[f"{who}_win365"] = _rate(p["win"], p["ran"], 0.07, 30.0)
        df[f"{who}_top3_365"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
        p = _by_date_prior(df, [col, "place", "surface"], ["ran", "top3"])
        df[f"{who}_course_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
        _reg("people", [f"{who}_n365", f"{who}_win365", f"{who}_top3_365", f"{who}_course_top3"])
    p = _by_date_prior(df, ["jockey_id", "trainer_id"], ["ran", "top3"])
    df["jt_n"] = p["ran"]
    df["jt_top3"] = _rate(p["top3"], p["ran"], 0.21, 15.0)
    _reg("people", ["elo_j", "elo_t", "jt_n", "jt_top3", "same_jockey"])

    # ---------- pedigree: progeny record of sire / broodmare sire under today's conditions
    for who in ("sire", "bms"):
        for name, keys in {"surf": [who, "surface"], "band": [who, "surface", "band"],
                           "wet": [who, "surface", "wet"]}.items():
            p = _by_date_prior(df, keys, ["ran", "win", "top3"])
            known = df[who].notna()
            df[f"{who}_{name}_top3"] = _rate(p["top3"], p["ran"], 0.21, 50.0).where(known)
            if name == "surf":
                df[f"{who}_{name}_win"] = _rate(p["win"], p["ran"], 0.07, 50.0).where(known)
                df[f"{who}_{name}_n"] = p["ran"].where(known)
                _reg("pedigree", [f"{who}_{name}_win", f"{who}_{name}_n"])
            _reg("pedigree", [f"{who}_{name}_top3"])
    _reg("pedigree", ["elo_sire", "elo_bms"])

    # ---------- race set-up: draw, pace, weights, field
    df["rel_number"] = (df["number"] - 1) / (df["n_runners"] - 1).clip(lower=1)
    df["draw_zone"] = pd.cut(df["rel_number"], [-0.01, 0.33, 0.66, 1.01], labels=False).astype(float)
    p = _by_date_prior(df, ["place", "surface", "distance", "draw_zone"], ["ran", "top3"])
    exp_top3 = (3.0 / df["n_runners"]).clip(upper=1.0)
    df["_exp3"] = exp_top3
    pe = _by_date_prior(df, ["place", "surface", "distance", "draw_zone"], ["_exp3"])
    df["draw_bias"] = ((p["top3"] + 20.0) / (pe["_exp3"] + 20.0)).where(df["draw_zone"].notna())
    # front-runner bias of the course: how well early leaders do there
    df["_front"] = (df["early_pos"] <= 0.25).astype(float).where(df["early_pos"].notna())
    df["_front_top3"] = df["_front"] * df["top3"]
    pf = _by_date_prior(df, ["place", "surface", "distance"], ["_front", "_front_top3"])
    df["course_front_top3"] = _rate(pf["_front_top3"], pf["_front"], 0.3, 30.0)
    gr = df.groupby("race_id", sort=False)
    df["_is_front"] = (df["style"] < 0.25).astype(float)
    df["_is_debut"] = (df["h_starts"] == 0).astype(float)
    gr = df.groupby("race_id", sort=False)
    df["pace_pressure"] = gr["_is_front"].transform("sum") / df["n_runners"]
    df["style_x_front"] = (0.5 - df["style"]) * (df["course_front_top3"] - 0.3)
    df["style_x_pressure"] = (0.5 - df["style"]) * df["pace_pressure"]
    df["is_turf"] = (df["surface"] == "芝").astype(float)
    df["place_code"] = df["place"].astype("category").cat.codes.astype(float)
    df["dir_code"] = df["direction"].map({"右": 0.0, "左": 1.0, "直": 2.0})
    _reg("race", ["number", "rel_number", "draw_bias", "style", "pace_pressure", "course_front_top3",
                  "style_x_front", "style_x_pressure", "weight_carried", "wc_diff", "bw_now", "wc_ratio",
                  "body_weight_diff", "n_runners", "distance", "is_turf", "going_code", "level",
                  "is_female", "is_gelding", "month", "place_code", "dir_code"])

    # ---------- relative-to-field versions of the strongest absolute features
    rel = ["elo_h", "elo_hs", "fig_max5", "fig_mean3", "l3f_mean3", "form_class5", "h_prize_per_start",
           "pos_mean3", "elo_j", "elo_t", "j_win365", "weight_carried", "h_top3_rate", "apt_band_top3",
           "elo_sire"]
    df = df.copy()  # defragment before adding many columns
    gr = df.groupby("race_id", sort=False)
    for c in rel:
        g = gr[c]
        mean, std = g.transform("mean"), g.transform("std")
        df[f"{c}_z"] = (df[c] - mean) / std.replace(0, np.nan)
        df[f"{c}_rk"] = g.rank(ascending=False, method="average") / df["n_runners"]
        grp = ("ability" if c in ("elo_h", "elo_hs", "fig_max5", "fig_mean3", "l3f_mean3", "form_class5",
                                  "h_prize_per_start", "h_top3_rate")
               else "form" if c == "pos_mean3"
               else "people" if c in ("elo_j", "elo_t", "j_win365")
               else "aptitude" if c == "apt_band_top3"
               else "pedigree" if c == "elo_sire" else "race")
        _reg(grp, [f"{c}_z", f"{c}_rk"])
    # debutants in the field (their ability is unknown to every feature above)
    df["field_debut_share"] = gr["_is_debut"].transform("mean")
    df["field_elo_max"] = gr["elo_h"].transform("max")
    df["elo_gap_to_best"] = df["field_elo_max"] - df["elo_h"]
    _reg("race", ["field_debut_share"])
    _reg("ability", ["elo_gap_to_best"])
    return df.drop(columns=["_exp3", "_front", "_front_top3", "_is_front", "_is_debut"])


def feature_columns() -> list[str]:
    cols: list[str] = []
    for names in FEATURE_GROUPS.values():
        for n in names:
            if n not in cols:
                cols.append(n)
    return cols


def build_features(raw: Path = RAW, card_races: pd.DataFrame | None = None,
                   card_runners: pd.DataFrame | None = None, pedigree_extra: pd.DataFrame | None = None
                   ) -> pd.DataFrame:
    races, results, _ = load_raw(raw)
    ped = load_pedigree(raw)
    if pedigree_extra is not None and len(pedigree_extra):
        ped = pd.concat([ped, pedigree_extra], ignore_index=True).drop_duplicates("horse_id")
    df = build_frame(races, results, ped, card_races, card_runners)
    return add_features(df)
