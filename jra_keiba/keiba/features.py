"""Load scraped data and build leak-free pre-race features.

Every feature of a runner uses only information available before the race:
earlier races of the horse / jockey / trainer / sire, the race conditions and
the draw. Upcoming race cards are appended to the history and go through the
same code path, so training and prediction features are computed identically.

Feature groups (the "aspects" reported with each prediction):
    ability    Elo ratings, speed figures, earnings, class reached
    form       recent finishes, margins, layoff
    aptitude   record on this surface / distance / course / going
    jockey     jockey rating, strike rates (year, last 30 days, course, distance, class),
               results against the market, riding style, record with this horse and stable,
               and what a change of jockey means
    trainer    trainer rating and strike rates (year, 30 days, distance, class, off a layoff or
               first time out, against the market), owner strike rates, travel east / west
    day        how the track has played in the earlier races of the same day (draw, pace)
    pedigree   sire and broodmare-sire progeny records
    race       pace set-up, draw, weights, field
"""
from __future__ import annotations

import warnings
from pathlib import Path

import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ID_COLS = {"race_id": str, "horse_id": str, "jockey_id": str, "trainer_id": str, "owner_id": str,
           "combo": str, "finish_raw": str}
GRADE_LEVEL = {"L": 5.5, "G3": 6.0, "G2": 7.0, "G1": 8.0, "重賞": 6.0}
CLASS_LEVEL = {"新馬": 1.0, "未勝利": 1.0, "1勝": 2.0, "2勝": 3.0, "3勝": 4.0, "OP": 5.0, "不明": 3.0}
GOING_CODE = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
NUMERIC_COLS = ["finish", "bracket", "number", "age", "weight_carried", "time_sec", "last3f", "win_odds",
                "popularity", "body_weight", "body_weight_diff", "prize"]


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
    # A card before the draw has no 馬番 / 馬体重 at all; an all-None column would turn the
    # merged column into dtype object and break the model input, so force numerics.
    for c in NUMERIC_COLS:
        if c in results.columns:
            results[c] = pd.to_numeric(results[c], errors="coerce").astype(float)
    races["level"] = class_level(races)
    keep = ["race_id", "date", "place", "surface", "direction", "inout", "distance", "going", "weather",
            "class", "grade", "level", "race_no", "name", "is_card", "cond", "laps"]
    df = results.merge(races[[c for c in keep if c in races.columns]], on="race_id", how="inner")
    df = df[df["surface"].isin(["芝", "ダ"])].copy()
    for c in ("owner_id", "trainer_region", "weather", "laps"):     # absent in minimal inputs
        if c not in df.columns:
            df[c] = np.nan

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
        df = df.merge(pedigree[["horse_id", "sire", "dam", "bms"]], on="horse_id", how="left")
    else:
        df["sire"] = np.nan
        df["dam"] = np.nan
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


# ------------------------------------------------------------------ runs outside JRA

NAR_TRACKS = {"門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知",
              "佐賀", "帯広", "旭川", "北見", "岩見沢", "上山", "三条", "足利", "宇都宮", "高崎", "益田", "福山", "荒尾",
              "中津"}
HORSE_RUNS_MIN_COVERAGE = 0.995


def load_horse_runs(raw: Path = RAW) -> pd.DataFrame:
    """Local-track and overseas runs (keiba.scrape horses). Empty frame if not collected."""
    p = raw / "horse_runs.csv.gz"
    cols = ["horse_id", "date", "venue", "n_runners", "finish", "popularity"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(p, dtype={"horse_id": str}, usecols=cols)


def horse_runs_usable(results: pd.DataFrame, raw: Path = RAW) -> tuple[bool, float]:
    """Outside runs may be used only when (almost) every horse has been read: horses are read
    newest first, so with partial coverage "has no outside runs" would partly mean "raced
    long ago" - the same trap as partial pedigree."""
    p = raw / "state.json"
    horses = results["horse_id"].dropna().unique()
    if not p.exists() or not len(horses):
        return False, 0.0
    done = set(json.loads(p.read_text()).get("horse_runs_done", []))
    cov = float(pd.Series(horses).isin(done).mean())
    return cov >= HORSE_RUNS_MIN_COVERAGE, cov


def add_outside_runs(df: pd.DataFrame, ext: pd.DataFrame | None) -> pd.DataFrame:
    """Features from a horse's runs outside JRA meetings *before* the race date: how many,
    how they went, whether the last one was abroad, and the true time since its last run.
    Without data every column is NaN."""
    cols = ["x_n", "x_n_since", "x_top3", "x_win", "x_over_n", "x_recent", "x_last_pos", "x_last_pop",
            "x_last_over", "days_since_any"]
    if ext is None or not len(ext):
        for c in cols:
            df[c] = np.nan
        return df
    e = ext.dropna(subset=["horse_id", "date"]).copy()
    e["date"] = pd.to_datetime(e["date"]).astype("datetime64[ns]")
    e = e[e["horse_id"].isin(set(df["horse_id"]))].sort_values(["horse_id", "date"], kind="stable")
    fin, n = pd.to_numeric(e["finish"], errors="coerce"), pd.to_numeric(e["n_runners"], errors="coerce")
    e["_pos"] = (fin - 1) / (n - 1).clip(lower=1)
    e["_top3"] = (fin <= 3).astype(float)
    e["_win"] = (fin == 1).astype(float)
    venue = e["venue"].astype(str)
    e["_over"] = (~venue.isin(NAR_TRACKS) & ~venue.str.contains("(地)", regex=False)).astype(float)
    g = e.groupby("horse_id", sort=False)
    e["_c_n"] = g.cumcount() + 1.0
    for c in ("_top3", "_win", "_over"):
        e[f"_c{c}"] = g[c].cumsum()
    e["_x_date"] = e["date"]
    e["_pop"] = pd.to_numeric(e["popularity"], errors="coerce")
    right = e[["horse_id", "date", "_x_date", "_c_n", "_c_top3", "_c_win", "_c_over", "_pos", "_pop", "_over"]] \
        .sort_values("date", kind="stable")
    left = df[["horse_id", "date"]].copy()
    left["date"] = left["date"].astype("datetime64[ns]")
    left["_row"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("date", kind="stable"), right, on="date", by="horse_id",
                      allow_exact_matches=False, direction="backward").sort_values("_row")
    m.index = df.index
    df["x_n"] = m["_c_n"].fillna(0.0)
    df["x_top3"] = _rate(m["_c_top3"].fillna(0.0), df["x_n"], 0.30, 3.0)
    df["x_win"] = _rate(m["_c_win"].fillna(0.0), df["x_n"], 0.10, 3.0)
    df["x_over_n"] = m["_c_over"].fillna(0.0)
    since_x = (left["date"].set_axis(df.index) - m["_x_date"]).dt.days.astype(float)
    # the last outside run counts as "recent" when it came after the horse's last JRA run
    recent = (since_x < df["days_since"]) | (df["days_since"].isna() & (df["x_n"] > 0))
    df["x_recent"] = recent.astype(float)
    df["x_last_pos"] = m["_pos"].where(recent)
    df["x_last_pop"] = m["_pop"].where(recent)
    df["x_last_over"] = m["_over"].where(recent)
    df["days_since_any"] = np.fmin(df["days_since"], since_x)
    prev = df.groupby("horse_id", sort=False)["x_n"].shift(1).fillna(0.0)
    df["x_n_since"] = (df["x_n"] - prev).clip(lower=0.0)
    return df


# ------------------------------------------------------------------ main feature builder

FEATURE_GROUPS: dict[str, list[str]] = {}


def _reg(group: str, names: list[str]) -> None:
    FEATURE_GROUPS.setdefault(group, [])
    for n in names:
        if n not in FEATURE_GROUPS[group]:
            FEATURE_GROUPS[group].append(n)


EAST = ("東京", "中山", "福島", "新潟")
WEST = ("京都", "阪神", "中京", "小倉")
WEATHER_CODE = {"晴": 0.0, "曇": 1.0, "小雨": 2.0, "雨": 3.0, "小雪": 4.0, "雪": 5.0}


def pace_balance(laps: str | None, distance: float) -> float:
    """First 600 m minus last 600 m of a race, in seconds (negative = fast early).
    The first lap is short when the distance is not a multiple of 200 m."""
    try:
        v = [float(x) for x in str(laps).split("-")]
    except ValueError:
        return np.nan
    if len(v) < 6 or not distance or np.isnan(distance):
        return np.nan
    first = distance % 200 or 200.0
    t, d = 0.0, 0.0
    for i, lap in enumerate(v):
        step = first if i == 0 else 200.0
        if d + step >= 600.0:
            t += lap * (600.0 - d) / step
            break
        t += lap
        d += step
    return t - sum(v[-3:])


def add_pace(df: pd.DataFrame) -> pd.DataFrame:
    """_pace_dev: pace balance of the race against the usual one for the surface and distance
    (median of earlier seasons; the first season uses itself). An outcome column: features
    read it only through a horse's earlier races."""
    r = df.drop_duplicates("race_id")[["race_id", "date", "surface", "distance"]].copy()
    laps = df.drop_duplicates("race_id")["laps"] if "laps" in df.columns else pd.Series(np.nan, index=r.index)
    r["bal"] = [pace_balance(l, d) for l, d in zip(laps, r["distance"])]
    r["y"] = r["date"].dt.year
    r["par"] = np.nan
    years = sorted(r["y"].unique())
    for y in years:
        cur = r["y"] == y
        base = (r["y"] < y) if y != years[0] else cur
        med = r[base & r["bal"].notna()].groupby(["surface", "distance"])["bal"].agg(["median", "size"])
        med = med[med["size"] >= 20]["median"]
        if len(med):
            r.loc[cur, "par"] = med.reindex(pd.MultiIndex.from_frame(r.loc[cur, ["surface", "distance"]])).to_numpy()
    r["_pace_dev"] = r["bal"] - r["par"]
    out = df.merge(r[["race_id", "_pace_dev"]], on="race_id", how="left")
    out.index = df.index
    return out


CLOSER_PRIOR = 0.30      # usual mean last-3F rank (0 = fastest in the race) of the first three


def day_bias(df: pd.DataFrame, day_races: pd.DataFrame | None = None) -> pd.DataFrame:
    """How the track has played today: where the first three of the *earlier* races of the
    same day, course and surface were drawn, and whether they were the fast finishers.

    Both are read from what the race-day result page shows at once: the draw, and the
    last-3F times (corner calls are posted only later, so they are not used here).
    ``day_races`` adds races that are not in the results yet (race day): columns date, place,
    surface, race_no, inner, closer - means over the first three of (number - 1) / (n - 1)
    and of the last-3F rank scaled to 0..1 (0 = fastest).
    Returns per-runner columns day_n, day_inner, day_closer (NaN before the first race)."""
    ran = df[(df["ran"] == 1) & df["last3f"].notna()]
    rk = ran.groupby("race_id")["last3f"].rank(method="average")
    cnt = ran.groupby("race_id")["last3f"].transform("size")
    frac = ((rk - 1) / (cnt - 1).clip(lower=1)).reindex(df.index)
    top = df.assign(_l3f_rank=frac)[(df["finish"] <= 3) & (df["ran"] == 1)]
    s = top.groupby("race_id").agg(inner=("rel_number", "mean"), closer=("_l3f_rank", "mean"))
    meta = df.drop_duplicates("race_id").set_index("race_id")[["date", "place", "surface", "_race_no"]]
    s = meta.join(s, how="left").reset_index().rename(columns={"_race_no": "race_no"})
    if day_races is not None and len(day_races):
        extra = day_races.copy()
        extra["date"] = pd.to_datetime(extra["date"])
        if "closer" not in extra.columns:
            extra["closer"] = np.nan
        for c in ("race_no", "inner", "closer"):      # closer can be None (no times posted yet)
            extra[c] = pd.to_numeric(extra[c], errors="coerce").astype(float)
        extra["race_id"] = [f"day{i}" for i in range(len(extra))]
        done = s[s["inner"].notna()]
        known = set(zip(done["date"], done["place"], done["race_no"]))
        extra = extra[[k not in known for k in zip(extra["date"], extra["place"], extra["race_no"])]]
        s = pd.concat([s, extra[list(s.columns)]], ignore_index=True)
    s = s.sort_values(["date", "place", "surface", "race_no"]).reset_index(drop=True)
    s[["inner", "closer"]] = s[["inner", "closer"]].astype(float)
    g = [s["date"], s["place"], s["surface"]]
    for c, prior in (("inner", 0.5), ("closer", CLOSER_PRIOR)):
        has = s[c].notna().astype(float)
        v = s[c].fillna(0.0)
        n = has.groupby(g).cumsum() - has
        tot = v.groupby(g).cumsum() - v
        s[f"day_{c}"] = ((tot + 2.0 * prior) / (n + 2.0)).where(n > 0)
        if c == "inner":
            s["day_n"] = n
    out = df[["race_id"]].merge(s[["race_id", "day_n", "day_inner", "day_closer"]], on="race_id", how="left")
    out.index = df.index
    return out[["day_n", "day_inner", "day_closer"]]


def add_features(df: pd.DataFrame, day_races: pd.DataFrame | None = None,
                 horse_runs: pd.DataFrame | None = None) -> pd.DataFrame:
    df = add_speed_figures(df)
    df = add_pace(df)
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
        _reg("jockey" if who == "j" else "trainer",
             [f"{who}_n365", f"{who}_win365", f"{who}_top3_365", f"{who}_course_top3"])
    p = _by_date_prior(df, ["jockey_id", "trainer_id"], ["ran", "top3"])
    df["jt_n"] = p["ran"]
    df["jt_top3"] = _rate(p["top3"], p["ran"], 0.21, 15.0)
    _reg("jockey", ["elo_j", "jt_n", "jt_top3", "same_jockey"])
    _reg("trainer", ["elo_t"])

    # ---------- jockey in depth (everything from earlier dates only; no odds of today's race)
    # hot or cold: the last 30 days against the jockey's own year
    p = _by_date_prior(df, ["jockey_id"], ["ran", "top3"], window="30D")
    df["j_n30"] = p["ran"]
    df["j_top3_30"] = _rate(p["top3"], p["ran"], 0.21, 15.0)
    df["j_hot"] = df["j_top3_30"] - df["j_top3_365"]
    # under today's conditions: distance band on this surface, and this class of race
    p = _by_date_prior(df, ["jockey_id", "surface", "band"], ["ran", "top3"])
    df["j_band_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
    df["_cls"] = pd.cut(df["level"], [-1, 1.5, 3.5, 5.2, 99], labels=False).astype(float)  # maiden / conditions / open / graded
    p = _by_date_prior(df, ["jockey_id", "_cls"], ["ran", "win", "top3"])
    df["j_class_n"] = p["ran"]
    df["j_class_win"] = _rate(p["win"], p["ran"], 0.07, 30.0)
    df["j_class_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
    # against the market: wins per win the odds implied over the past year (1.0 = as expected).
    # Uses the odds of *earlier* races only, so the odds-free model stays odds-free today.
    inv = 1.0 / df["win_odds"].where(df["win_odds"] > 0)
    df["_mkt_p"] = inv / inv.groupby(df["race_id"]).transform("sum")
    df["_win_o"] = df["win"].where(df["_mkt_p"].notna())
    p = _by_date_prior(df, ["jockey_id"], ["_win_o", "_mkt_p"], window="365D")
    df["j_ae_win"] = (p["_win_o"] + 10.0) / (p["_mkt_p"] + 10.0)
    # riding style: where the jockey's mounts sit early, and how that fits this horse
    df["_ep"] = df["early_pos"]
    df["_ep_n"] = df["early_pos"].notna().astype(float)
    p = _by_date_prior(df, ["jockey_id"], ["_ep", "_ep_n"], window="365D")
    df["j_early"] = (p["_ep"] + 10.0 * 0.5) / (p["_ep_n"] + 10.0)
    df["j_style_gap"] = df["j_early"] - df["style"]          # negative: rides more forward than the horse has run
    # this jockey on this horse
    df["jh_n"] = _prior_sum(df, ["horse_id", "jockey_id"], "ran")
    df["jh_top3"] = _rate(_prior_sum(df, ["horse_id", "jockey_id"], "top3"), df["jh_n"], 0.24, 2.0)
    df["jh_win"] = _rate(_prior_sum(df, ["horse_id", "jockey_id"], "win"), df["jh_n"], 0.08, 2.0)
    df["jh_share"] = df["jh_n"] / df["h_starts"].clip(lower=1)      # how much of the horse's career
    # change of jockey: is today's jockey rated above the last one (ratings as of each race)?
    prev_elo_j = gh["elo_j"].shift(1)
    df["j_upgrade"] = (df["elo_j"] - prev_elo_j).where(prev_jockey.notna()) * (1.0 - df["same_jockey"].fillna(1.0))
    # how much this stable uses this jockey (share of the stable's runners over a year)
    p = _by_date_prior(df, ["jockey_id", "trainer_id"], ["ran"], window="365D")
    df["jt_share365"] = (p["ran"] / df["t_n365"].clip(lower=1.0)).clip(upper=1.0)
    _reg("jockey", ["j_n30", "j_top3_30", "j_hot", "j_band_top3", "j_class_n", "j_class_win", "j_class_top3",
                    "j_ae_win", "j_early", "j_style_gap", "jh_n", "jh_top3", "jh_win", "jh_share",
                    "j_upgrade", "jt_share365"])

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

    # ---------- trainer in depth, owner, travel
    card_owner = df["is_card"] & df["owner_id"].isna()     # cards do not list the owner: last known one
    df.loc[card_owner, "owner_id"] = df.groupby("horse_id", sort=False)["owner_id"].ffill()[card_owner]
    p = _by_date_prior(df, ["trainer_id"], ["ran", "top3"], window="30D")
    df["t_n30"] = p["ran"]
    df["t_top3_30"] = _rate(p["top3"], p["ran"], 0.21, 15.0)
    df["t_hot"] = df["t_top3_30"] - df["t_top3_365"]
    p = _by_date_prior(df, ["trainer_id", "surface", "band"], ["ran", "top3"])
    df["t_band_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
    p = _by_date_prior(df, ["trainer_id", "_cls"], ["ran", "win", "top3"])
    df["t_class_win"] = _rate(p["win"], p["ran"], 0.07, 30.0)
    df["t_class_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
    p = _by_date_prior(df, ["trainer_id"], ["_win_o", "_mkt_p"], window="365D")
    df["t_ae_win"] = (p["_win_o"] + 10.0) / (p["_mkt_p"] + 10.0)
    # how the stable does in today's situation: first start / back from 90+ days off / in training
    df["_sit"] = np.select([df["h_starts"] == 0, df["days_since"] >= 90], [2.0, 1.0], 0.0)
    p = _by_date_prior(df, ["trainer_id", "_sit"], ["ran", "win", "top3"])
    df["t_sit_win"] = _rate(p["win"], p["ran"], 0.07, 20.0)
    df["t_sit_top3"] = _rate(p["top3"], p["ran"], 0.21, 20.0)
    df["layoff"] = (df["_sit"] == 1.0).astype(float)
    # travel: a Miho (east) horse at a western course or the reverse. Hokkaido is away for both.
    zone = df["place"].map({**{p_: "東" for p_ in EAST}, **{p_: "西" for p_ in WEST}})
    home = df["trainer_region"].isin(["東", "西"])
    df["away"] = ((zone != df["trainer_region"]) | zone.isna()).astype(float).where(home)
    df["region_west"] = (df["trainer_region"] == "西").astype(float).where(home)
    df["_away_k"] = df["away"].fillna(-1.0)
    p = _by_date_prior(df, ["trainer_id", "_away_k"], ["ran", "top3"])
    df["t_away_top3"] = _rate(p["top3"], p["ran"], 0.21, 30.0)
    st = _prior_sum(df, ["horse_id", "_away_k"], "ran")
    df["apt_away_n"] = st
    df["apt_away_top3"] = _rate(_prior_sum(df, ["horse_id", "_away_k"], "top3"), st, 0.24, 2.0)
    p = _by_date_prior(df, ["owner_id"], ["ran", "win", "top3"], window="365D")
    known = df["owner_id"].notna()
    df["o_n365"] = p["ran"].where(known)
    df["o_win365"] = _rate(p["win"], p["ran"], 0.07, 30.0).where(known)
    df["o_top3_365"] = _rate(p["top3"], p["ran"], 0.21, 30.0).where(known)
    _reg("trainer", ["t_n30", "t_top3_30", "t_hot", "t_band_top3", "t_class_win", "t_class_top3", "t_ae_win",
                     "t_sit_win", "t_sit_top3", "layoff", "away", "region_west", "t_away_top3",
                     "o_n365", "o_win365", "o_top3_365"])
    _reg("aptitude", ["apt_away_n", "apt_away_top3"])

    # ---------- dam: how her other foals have done (this horse's own runs taken out)
    p = _by_date_prior(df, ["dam"], ["ran", "win", "top3", "prize"])
    known = df["dam"].notna()
    dam_n = (p["ran"] - df["h_starts"]).clip(lower=0.0)
    df["dam_n"] = dam_n.where(known)
    df["dam_win"] = _rate((p["win"] - df["h_win_sum"]).clip(lower=0.0), dam_n, 0.07, 10.0).where(known)
    df["dam_top3"] = _rate((p["top3"] - df["h_top3_sum"]).clip(lower=0.0), dam_n, 0.21, 10.0).where(known)
    df["dam_prize"] = np.log1p((p["prize"] - df["h_prize_sum"]).clip(lower=0.0) / dam_n.clip(lower=1.0)).where(known)
    _reg("pedigree", ["dam_n", "dam_win", "dam_top3", "dam_prize"])

    # ---------- pace of the horse's earlier races, and how it copes with fast and slow ones
    gp = df.groupby("horse_id", sort=False)["_pace_dev"]
    l1, l2, l3 = gp.shift(1), gp.shift(2), gp.shift(3)
    df["pace_dev_l1"] = l1
    cnt = l1.notna().astype(float) + l2.notna() + l3.notna()
    df["pace_dev_mean3"] = (l1.fillna(0) + l2.fillna(0) + l3.fillna(0)) / cnt.replace(0, np.nan)
    for name, mask in (("fast", df["_pace_dev"] < -0.5), ("slow", df["_pace_dev"] > 0.5)):
        df[f"_{name}"] = mask.astype(float)
        df[f"_{name}_top3"] = df[f"_{name}"] * df["top3"].fillna(0.0)
        n = _prior_sum(df, ["horse_id"], f"_{name}")
        df[f"apt_{name}_n"] = n
        df[f"apt_{name}_top3"] = _rate(_prior_sum(df, ["horse_id"], f"_{name}_top3"), n, 0.24, 2.0)
    df["pace_fit"] = (df["apt_fast_top3"] - df["apt_slow_top3"]) * (df["pace_pressure"] - 0.25)
    df["weather_code"] = df["weather"].map(WEATHER_CODE)       # 0 = fine; unknown on early cards
    _reg("race", ["pace_dev_l1", "pace_dev_mean3", "apt_fast_n", "apt_fast_top3", "apt_slow_n", "apt_slow_top3",
                  "pace_fit", "weather_code"])

    # ---------- runs outside JRA (local tracks, abroad) before this race
    df = add_outside_runs(df, horse_runs)
    _reg("form", ["x_n", "x_n_since", "x_top3", "x_win", "x_over_n", "x_recent", "x_last_pos", "x_last_pop",
                  "x_last_over", "days_since_any"])

    # ---------- today's track: where the first three of the earlier races were drawn, and
    # whether they were the fast finishers
    df["_race_no"] = pd.to_numeric(df["race_id"].str[-2:], errors="coerce")
    b = day_bias(df, day_races)
    df["day_n"], df["day_inner"] = b["day_n"], b["day_inner"] - 0.5
    df["day_closer"] = CLOSER_PRIOR - b["day_closer"]                  # positive: fast finishers are winning today
    df["day_inner_x"] = (df["rel_number"] - 0.5) * df["day_inner"]     # positive: drawn on the side that is winning
    df["day_closer_x"] = (df["style"] - 0.42) * df["day_closer"]       # positive: runs the way that is winning
    df["day_l3f_x"] = df["l3f_mean3"] * df["day_closer"]               # a strong finisher on a finishers' day
    _reg("day", ["day_n", "day_inner", "day_closer", "day_inner_x", "day_closer_x", "day_l3f_x"])

    # ---------- relative-to-field versions of the strongest absolute features
    rel = ["elo_h", "elo_hs", "fig_max5", "fig_mean3", "l3f_mean3", "form_class5", "h_prize_per_start",
           "pos_mean3", "elo_j", "elo_t", "j_win365", "weight_carried", "h_top3_rate", "apt_band_top3",
           "elo_sire", "j_ae_win", "j_class_top3"]
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
               else "jockey" if c in ("elo_j", "j_win365", "j_ae_win", "j_class_top3")
               else "trainer" if c == "elo_t"
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


PEDIGREE_MIN_COVERAGE = 0.995


def pedigree_usable(results: pd.DataFrame, ped: pd.DataFrame) -> tuple[bool, float]:
    """Pedigree may only be used when (almost) every horse in the results has it.

    It is collected starting from the most recent race cards, so while coverage is
    partial, "has pedigree" means "was still racing later" - a look-ahead signal
    (in 2024 maiden races: 20% winners with pedigree vs 3% without). Partial pedigree
    is therefore dropped entirely rather than left for the model to exploit.
    """
    horses = results["horse_id"].dropna().unique()
    if not len(horses) or not len(ped):
        return False, 0.0
    cov = float(pd.Series(horses).isin(ped["horse_id"]).mean())
    return cov >= PEDIGREE_MIN_COVERAGE, cov


def build_features(raw: Path = RAW, card_races: pd.DataFrame | None = None,
                   card_runners: pd.DataFrame | None = None, pedigree_extra: pd.DataFrame | None = None,
                   log=print, use_pedigree: bool | None = None,
                   day_races: pd.DataFrame | None = None, use_horse_runs: bool | None = None,
                   horse_runs_extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """``use_pedigree=False`` keeps pedigree out even when coverage would allow it: a model
    trained without pedigree must be fed features built without it. ``df.attrs['pedigree']``
    records what was done. ``use_horse_runs`` / ``df.attrs['horse_runs']`` do the same for
    the runs outside JRA; ``horse_runs_extra`` adds freshly read runs of the card's horses."""
    races, results, _ = load_raw(raw)
    ped = load_pedigree(raw)
    ok, cov = pedigree_usable(results, ped)
    if use_pedigree is False:
        ped = None
    elif not ok:
        if len(ped):
            log(f"pedigree covers only {cov:.1%} of horses -> not used (partial coverage leaks the future)")
        ped = None
    elif pedigree_extra is not None and len(pedigree_extra):
        ped = pd.concat([ped, pedigree_extra], ignore_index=True).drop_duplicates("horse_id")
    hr = load_horse_runs(raw)
    ok, cov = horse_runs_usable(results, raw)
    if use_horse_runs is False:
        hr = None
    elif not ok:
        if len(hr):
            log(f"runs outside JRA are read for only {cov:.1%} of horses -> not used")
        hr = None
    elif horse_runs_extra is not None and len(horse_runs_extra):
        hr = pd.concat([hr, horse_runs_extra[[c for c in hr.columns if c in horse_runs_extra.columns]]],
                       ignore_index=True)
        hr["date"] = pd.to_datetime(hr["date"]).dt.strftime("%Y-%m-%d")
        hr = hr.drop_duplicates(["horse_id", "date"], keep="last")
    out = add_features(build_frame(races, results, ped, card_races, card_runners), day_races, hr)
    out.attrs["pedigree"] = ped is not None
    out.attrs["horse_runs"] = hr is not None
    return out
