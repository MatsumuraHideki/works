"""Features of a race must not depend on that race's (or any later race's) outcome."""
import numpy as np
import pandas as pd

from keiba import features


def _toy(seed: int = 0, n_days: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    races, rows = [], []
    for d in range(n_days):
        date = pd.Timestamp("2023-09-02") + pd.Timedelta(days=7 * d)
        pool = rng.permutation(80)          # a horse runs at most once a day
        for rn in range(1, 7):
            rid = f"{date.year}{5 + rn % 2:02d}{d // 12 + 1:02d}{d % 12 + 1:02d}{rn:02d}"
            n = int(rng.integers(6, 12))
            races.append(dict(race_id=rid, date=date, place="東京" if rn % 2 else "中山", surface="芝" if rn % 3 else "ダ",
                              direction="左", inout=None, distance=int(rng.choice([1400, 1600, 2000])),
                              going=rng.choice(["良", "重"]), weather=rng.choice(["晴", "雨"]), grade=None,
                              race_no=rn, name="t", cond="",
                              laps="-".join(f"{x:.1f}" for x in rng.uniform(11.5, 13.0, 8)), **{"class": "1勝"}))
            horses, pool = pool[:n], pool[n:]
            fin = rng.permutation(n) + 1
            for i in range(n):
                rows.append(dict(race_id=rid, finish=float(fin[i]), finish_raw=str(fin[i]), number=i + 1.0,
                                 horse_id=f"h{horses[i]}", horse_name="x", sex="牡", age=4.0, weight_carried=56.0,
                                 jockey_id=f"j{rng.integers(15)}", trainer_id=f"t{horses[i] % 20}",
                                 owner_id=f"o{horses[i] % 7}", trainer_region="東" if horses[i] % 2 else "西",
                                 time_sec=95.0 + fin[i] * 0.2, passing=f"{fin[i]}-{fin[i]}", last3f=35.0,
                                 win_odds=float(rng.uniform(2, 50)), popularity=float(i + 1), body_weight=480.0,
                                 body_weight_diff=0.0, prize=float(max(0, 4 - fin[i]) * 100)))
    ped = pd.DataFrame({"horse_id": [f"h{i}" for i in range(80)], "sire": [f"s{i % 9}" for i in range(80)],
                        "dam": [f"d{i % 30}" for i in range(80)], "bms": [f"b{i % 5}" for i in range(80)]})
    return pd.DataFrame(races), pd.DataFrame(rows), ped


def test_no_leak_from_current_or_future_results():
    races, results, ped = _toy()
    base = features.add_features(features.build_frame(races, results, ped))
    cols = features.feature_columns()
    cutoff = races["date"].sort_values().unique()[-6]
    late = races.loc[races["date"] >= cutoff, "race_id"]
    changed = results.copy()
    m = changed["race_id"].isin(late)
    rng = np.random.default_rng(1)
    # rewrite every outcome column from the cutoff date on
    changed.loc[m, "finish"] = changed[m].groupby("race_id")["finish"].transform(lambda s: rng.permutation(s.to_numpy()))
    changed.loc[m, "finish_raw"] = changed.loc[m, "finish"].astype(int).astype(str)
    changed.loc[m, "time_sec"] += rng.normal(0, 3, m.sum())
    changed.loc[m, "last3f"] += rng.normal(0, 2, m.sum())
    changed.loc[m, "prize"] = 999.0
    changed.loc[m, "passing"] = "1-1"
    changed.loc[m, "win_odds"] = 3.3
    changed.loc[m, "popularity"] = 1.0
    races2 = races.copy()                    # lap times are an outcome too
    races2.loc[races2["race_id"].isin(late), "laps"] = "12.0-12.0-12.0-12.0-12.0-12.0-12.0-12.0"
    alt = features.add_features(features.build_frame(races2, changed, ped))
    first_day = races.loc[races["date"] == cutoff, "race_id"]
    # the "day" features read the earlier races of the same day by design: tested separately below
    day = set(features.FEATURE_GROUPS["day"])
    a = base[base["race_id"].isin(first_day)].set_index(["race_id", "horse_id"])[cols].sort_index()
    b = alt[alt["race_id"].isin(first_day)].set_index(["race_id", "horse_id"])[cols].sort_index()
    diff = [c for c in cols if c not in day
            and not np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-9)]
    assert diff == [], f"features leaking outcomes: {diff}"
    assert len(cols) > 80


def test_day_bias_reads_only_earlier_races_of_the_day():
    """Today's-track features of a race may use the races before it on the day, never the
    race itself or later ones - and they must exist once an earlier race has been run."""
    races, results, ped = _toy()
    base = features.add_features(features.build_frame(races, results, ped))
    cols = features.feature_columns()
    cutoff = races["date"].sort_values().unique()[-6]
    late = races.loc[(races["date"] > cutoff) | ((races["date"] == cutoff) & (races["race_no"] >= 4)), "race_id"]
    changed = results.copy()
    m = changed["race_id"].isin(late)
    rng = np.random.default_rng(2)
    changed.loc[m, "finish"] = changed[m].groupby("race_id")["finish"].transform(lambda s: rng.permutation(s.to_numpy()))
    changed.loc[m, "finish_raw"] = changed.loc[m, "finish"].astype(int).astype(str)
    changed.loc[m, "passing"] = "1-1"
    alt = features.add_features(features.build_frame(races, changed, ped))
    upto = races.loc[(races["date"] == cutoff) & (races["race_no"] <= 4), "race_id"]     # race 4 itself included
    a = base[base["race_id"].isin(upto)].set_index(["race_id", "horse_id"])[cols].sort_index()
    b = alt[alt["race_id"].isin(upto)].set_index(["race_id", "horse_id"])[cols].sort_index()
    diff = [c for c in cols if not np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-9)]
    assert diff == [], f"features reading the race itself or later races: {diff}"
    last = base[base["race_id"].isin(races.loc[(races["date"] == cutoff) & (races["race_no"] >= 5), "race_id"])]
    assert last["day_n"].max() >= 1 and last.loc[last["day_n"] >= 1, "day_inner"].notna().all()


def test_card_rows_get_features_without_results():
    races, results, ped = _toy()
    last = races["race_id"].iloc[-1]
    card_r = races[races["race_id"] == last].assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
    card_x = results[results["race_id"] == last].copy()
    for c in ("finish", "time_sec", "last3f", "win_odds", "popularity", "prize"):
        card_x[c] = np.nan
    card_x["finish_raw"], card_x["passing"] = "", None
    hist_r, hist_x = races[races["race_id"] != last], results[results["race_id"] != last]
    full = features.add_features(features.build_frame(races, results, ped))
    card = features.add_features(features.build_frame(hist_r, hist_x, ped, card_r, card_x))
    cols = features.feature_columns()
    a = full[full["race_id"] == last].set_index("horse_id")[cols].sort_index().astype(float)
    b = card[card["race_id"] == last].set_index("horse_id")[cols].sort_index().astype(float)
    diff = [c for c in cols if not np.allclose(a[c], b[c], equal_nan=True, atol=1e-9)]
    assert diff == [], f"card features differ from historical features: {diff}"


def test_partial_pedigree_is_not_used():
    races, results, ped = _toy()
    assert features.pedigree_usable(results, ped) == (True, 1.0)
    ok, cov = features.pedigree_usable(results, ped.iloc[: len(ped) // 2])
    assert not ok and 0.4 < cov < 0.6
    assert features.pedigree_usable(results, ped.iloc[:0]) == (False, 0.0)


def test_pre_draw_card_without_weights_or_numbers_stays_numeric():
    races, results, ped = _toy()
    last = races["race_id"].iloc[-1]
    card_r = races[races["race_id"] == last].assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
    card_x = results[results["race_id"] == last].copy().astype(object)
    for c in ("finish", "time_sec", "last3f", "win_odds", "popularity", "prize", "number", "body_weight",
              "body_weight_diff"):
        card_x[c] = None                      # what a card looks like before the draw
    card_x["finish_raw"], card_x["passing"] = "", None
    hist_r, hist_x = races[races["race_id"] != last], results[results["race_id"] != last]
    df = features.add_features(features.build_frame(hist_r, hist_x, ped, card_r, card_x))
    cols = features.feature_columns()
    bad = [c for c in cols if df[c].dtype.kind not in "fiub"]
    assert bad == [], f"non-numeric feature columns: {bad}"
    assert df["is_card"].sum() == len(card_x)


def test_day_races_without_times_are_accepted():
    """Race day: a finished race whose last-3F times are not posted yet still counts for the draw."""
    races, results, ped = _toy()
    last = races["date"].max()
    day = pd.DataFrame([{"race_id": "x", "date": last, "place": "東京", "surface": "芝", "race_no": 0,
                         "inner": 0.2, "closer": None}])
    out = features.add_features(features.build_frame(races, results, ped), day)
    first = out[(out["date"] == last) & (out["place"] == "東京") & (out["surface"] == "芝")].sort_values("race_id")
    row = first.iloc[0]
    assert row["day_n"] == 1 and abs(row["day_inner"] - ((0.2 + 2 * 0.5) / 3 - 0.5)) < 1e-9
    assert np.isnan(row["day_closer"])


def _ext(races, results, seed=5):
    """Toy runs outside JRA: a few per horse, on days between the meetings."""
    rng = np.random.default_rng(seed)
    days = sorted(races["date"].unique())
    rows = []
    for h in results["horse_id"].unique()[::3]:
        for d in rng.choice(len(days) - 1, size=3, replace=False):
            rows.append(dict(horse_id=h, date=(pd.Timestamp(days[d]) + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                             venue=rng.choice(["大井", "園田", "シャティン"]), n_runners=12.0,
                             finish=float(rng.integers(1, 13)), popularity=float(rng.integers(1, 13))))
    return pd.DataFrame(rows)


def test_outside_runs_are_read_only_before_the_race():
    """Runs outside JRA on or after a race date must not move that race's features, and an
    earlier one must show up in the counts and in the true days since the last run."""
    races, results, ped = _toy()
    ext = _ext(races, results)
    frame = features.build_frame(races, results, ped)
    base = features.add_features(frame.copy(), None, ext)
    cols = features.FEATURE_GROUPS["form"]
    cutoff = pd.Timestamp(races["date"].sort_values().unique()[-6])
    changed = ext.copy()
    late = pd.to_datetime(changed["date"]) >= cutoff
    changed.loc[late, ["finish", "popularity"]] = 1.0
    changed = pd.concat([changed, changed[late].assign(date=cutoff.strftime("%Y-%m-%d"), venue="香港")])
    alt = features.add_features(frame.copy(), None, changed)
    a = base[base["date"] <= cutoff].set_index(["race_id", "horse_id"])[cols].sort_index()
    b = alt[alt["date"] <= cutoff].set_index(["race_id", "horse_id"])[cols].sort_index()
    diff = [c for c in cols if not np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-9)]
    assert diff == [], f"features reading later outside runs: {diff}"
    # one horse checked by hand
    h = ext["horse_id"].iloc[0]
    row = base[base["horse_id"] == h].sort_values("date").iloc[-1]
    before = ext[(ext["horse_id"] == h) & (pd.to_datetime(ext["date"]) < row["date"])]
    assert row["x_n"] == len(before)
    if len(before):
        gap = (row["date"] - pd.to_datetime(before["date"]).max()).days
        assert row["days_since_any"] <= gap
    none = features.add_features(frame.copy(), None, None)
    assert none["x_n"].isna().all()
