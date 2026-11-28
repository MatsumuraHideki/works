"""Fetch upcoming race cards and odds, and store them as JSON snapshots.

A snapshot (``data/cards/<race_id>.json``) holds everything prediction needs from
the network, so predictions can be re-run offline from the committed file.

Usage:
    python -m keiba.cards [--dates 20261114,20261115] [--grades G1,G2,G3 | --grades all]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import netkeiba as nk
from .http import Fetcher, decode
from .scrape import Blocked, fetch_text, today_jst

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "cards"
ODDS_TYPES = (1, 4, 5, 6, 7, 8)  # the type-1 response also carries 複勝


def upcoming_dates(f: Fetcher, days_ahead: int = 6) -> list[str]:
    today = today_jst()
    limit = today + dt.timedelta(days=days_ahead)
    dates: list[str] = []
    for y, m in sorted({(today.year, today.month), (limit.year, limit.month)}):
        dates += nk.parse_calendar(fetch_text(f, nk.url_calendar(y, m)))
    lo, hi = today.strftime("%Y%m%d"), limit.strftime("%Y%m%d")
    return sorted(d for d in set(dates) if lo <= d <= hi)


def fetch_card(f: Fetcher, race_id: str, listed: dict | None = None) -> dict | None:
    card = nk.parse_shutuba(fetch_text(f, nk.url_shutuba(race_id)), race_id)
    if card is None or not card["runners"]:
        return None
    if listed and not card["race"].get("grade"):
        card["race"]["grade"] = listed.get("grade")
    card["pedigree"] = nk.parse_shutuba_past_pedigree(fetch_text(f, nk.url_shutuba_past(race_id)))
    odds: dict[str, dict] = {}
    status = None
    # While sales are open, "init" serves a snapshot that can be ~50 minutes old and "update"
    # the current board; before sales (predicted odds) and after the race only "init" has
    # data. The action is chosen once, on the 単勝 table, and used for every bet type so one
    # card never mixes a live table with a stale one.
    action = "update"
    for t in ODDS_TYPES:
        text = fetch_text(f, nk.url_odds(race_id, t).replace("action=init", f"action={action}"),
                          referer=nk.url_shutuba(race_id))
        if t == 1 and not nk.parse_odds(text, 1)["odds"]:
            action = "init"
            text = fetch_text(f, nk.url_odds(race_id, t), referer=nk.url_shutuba(race_id))
        for tt in ((1, 2) if t == 1 else (t,)):
            o = nk.parse_odds(text, tt)
            if tt == 1:
                status = o["status"]
                card["odds_time"] = o["time"]
            if o["odds"]:
                odds[nk.ODDS_TYPE[tt]] = {"-".join(map(str, k)): list(v) for k, v in o["odds"].items()}
        if status in ("NG", "yoso", None):
            break  # no real odds yet: other bet types are not served either
    card["odds_action"] = action
    card["odds_status"] = status
    card["odds"] = odds
    card["fetched_at"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")
    return card


HORSE_CACHE_DAYS = 3


def horse_runs(f: Fetcher, card: dict) -> list[dict]:
    """Runs outside JRA of the card's horses, read from each horse's result page so that a
    recent local or overseas run is known on race day. One request per horse, cached for a
    few days in data/cards/horses/ (the first fetch of a card takes about a minute)."""
    cache = CARDS / "horses"
    cache.mkdir(parents=True, exist_ok=True)
    today = today_jst()
    out = []
    for r in card["runners"]:
        hid = r.get("horse_id")
        if not hid or r.get("cancelled"):
            continue
        p = cache / f"{hid}.json"
        if p.exists():
            saved = json.loads(p.read_text())
            if (today - dt.date.fromisoformat(saved["fetched"])).days <= HORSE_CACHE_DAYS:
                out += saved["runs"]
                continue
        runs = nk.parse_horse_runs(fetch_text(f, nk.url_horse_result(hid)), hid)
        if runs is None:
            continue
        keep = [{k: g[k] for k in ("horse_id", "date", "venue", "n_runners", "finish", "popularity")}
                for g in runs if not g["jra"]]
        p.write_text(json.dumps({"fetched": today.isoformat(), "runs": keep}, ensure_ascii=False))
        out += keep
    return out


def day_races(f: Fetcher, race_id: str, listed: list[dict], card: dict) -> list[dict]:
    """Track-bias summaries of the races already run today at this course (for the model's
    "today's track" features). One request per finished race, cached in data/cards/day/."""
    out = []
    cache = CARDS / "day"
    cache.mkdir(parents=True, exist_ok=True)
    same = [r for r in listed if r["race_id"][:10] == race_id[:10] and int(r["race_id"][10:]) < int(race_id[10:])]
    print(f"  day races for {race_id}: {len(same)} earlier at the course, "
          f"{sum(r['is_result'] for r in same)} with a result link", flush=True)
    for r in same:
        rid = r["race_id"]
        if not r["is_result"]:
            continue
        p = cache / f"{rid}.json"
        if p.exists():
            row = json.loads(p.read_text())
        else:
            res = nk.parse_result_page(fetch_text(f, nk.url_result(rid)), rid)
            bias = nk.day_bias_of_result(res) if res else None
            if res is None or bias is None or res.get("surface") not in ("芝", "ダ"):
                why = ("no payouts on the page" if res is None else f"surface {res.get('surface')}"
                       if res.get("surface") not in ("芝", "ダ") else f"order rows {len(res['order'])}, "
                       f"first {res['order'][:1]}")
                print(f"  day race {rid}: skipped ({why})", flush=True)
                continue
            if bias["closer"] is None:
                print(f"  day race {rid}: no last-3F times yet (first row {res['order'][:1]})", flush=True)
            row = {"race_id": rid, "date": card["race"]["date"], "place": card["race"]["place"],
                   "surface": res["surface"], "race_no": int(rid[10:]), **bias}
            if bias["closer"] is not None:    # complete: no need to read this race again
                p.write_text(json.dumps(row, ensure_ascii=False))
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", default="", help="comma separated YYYYMMDD; default: race days in the next 6 days")
    ap.add_argument("--grades", default="G1,G2,G3", help="comma separated grades, or 'all'")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--results", default="", help="comma separated race ids: save their result and payouts instead")
    args = ap.parse_args(argv)
    f = Fetcher(min_interval=args.interval)
    if args.results:
        out = CARDS / "results"
        out.mkdir(parents=True, exist_ok=True)
        for rid in [x for x in args.results.split(",") if x]:
            res = nk.parse_result_page(fetch_text(f, nk.url_result(rid)), rid)
            if res is None:
                print(f"result {rid}: not published yet")
                continue
            try:        # final 単勝 odds, to measure how far the odds moved after the prediction
                o = nk.parse_odds(fetch_text(f, nk.url_odds(rid, 1), referer=nk.url_result(rid)), 1)
                if o["status"] == "result":
                    res["final_win_odds"] = {str(k[0]): v[0] for k, v in o["odds"].items()}
            except Exception as exc:  # the result itself is what matters; odds are a bonus
                print(f"result {rid}: final odds not saved ({exc})")
            (out / f"{rid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print(f"result {rid}: {len(res['payouts'])} payouts, winner {res['order'][0] if res['order'] else '?'}")
        return 0
    try:
        dates = [d for d in args.dates.split(",") if d] or upcoming_dates(f)
        grades = None if args.grades == "all" else set(args.grades.split(","))
        CARDS.mkdir(parents=True, exist_ok=True)
        n = 0
        for date in dates:
            listed = nk.parse_race_list_sub(fetch_text(f, nk.url_race_list_sub(date)))
            for r in listed:
                if r["is_result"] or (grades is not None and r["grade"] not in grades):
                    continue
                card = fetch_card(f, r["race_id"], r)
                if card is None:
                    continue
                card["race"]["date"] = card["race"]["date"] or f"{date[:4]}-{date[4:6]}-{date[6:]}"
                try:        # earlier races of the day at this course; never worth losing the card for
                    card["day_races"] = day_races(f, r["race_id"], listed, card)
                except Blocked:
                    raise
                except Exception as exc:
                    print(f"card {r['race_id']}: earlier races of the day not read ({exc})", flush=True)
                    card["day_races"] = []
                try:        # recent runs outside JRA of the runners
                    card["horse_runs"] = horse_runs(f, card)
                except Blocked:
                    raise
                except Exception as exc:
                    print(f"card {r['race_id']}: runs outside JRA not read ({exc})", flush=True)
                    card["horse_runs"] = []
                (CARDS / f"{r['race_id']}.json").write_text(json.dumps(card, ensure_ascii=False, indent=1))
                n += 1
                print(f"card {r['race_id']} {card['race']['date']} {card['race']['name']} "
                      f"runners={len(card['runners'])} odds={card['odds_status']} "
                      f"types={list(card['odds'])} day_races={len(card['day_races'])}", flush=True)
        print(f"{n} cards saved for dates {dates}")
    except Blocked as exc:
        print(f"BLOCKED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
