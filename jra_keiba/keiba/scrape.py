"""Incremental scraper for JRA race results (all classes, graded races included).

Storage layout (under ``data/raw``), one immutable file per month:
    races/YYYYMM.csv.gz     one row per race
    results/YYYYMM.csv.gz   one row per runner
    payouts/YYYYMM.csv.gz   one row per winning ticket
    odds/YYYYMM.csv.gz      final odds of every ticket for graded races (optional)
    pedigree.csv.gz         horse_id -> sire / dam / broodmare sire
    state.json              months that are complete

Usage:
    python -m keiba.scrape results --from 2026-11 --to 2022-01 [--max-minutes 330] [--push]
    python -m keiba.scrape odds --from 2026-11 --to 2023-01 [--grades G1,G2,G3]
    python -m keiba.scrape pedigree [--max-minutes 60]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import netkeiba as nk
from .http import Fetcher, decode

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
JST = dt.timezone(dt.timedelta(hours=9))


class Blocked(RuntimeError):
    """The site refused requests; stop instead of hammering it."""


def today_jst() -> dt.date:
    return dt.datetime.now(JST).date()


def months_desc(start: str, end: str) -> list[str]:
    """'2026-11','2022-01' -> ['202611', ..., '202201'] (newest first)."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) >= (ey, em):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def load_state() -> dict:
    p = RAW / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"complete_months": [], "odds_done": [], "ped_races_done": []}


def save_state(state: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for k in state:
        if isinstance(state[k], list):
            state[k] = sorted(set(state[k]))
    (RAW / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=0))


def read_month(kind: str, ym: str) -> pd.DataFrame:
    p = RAW / kind / f"{ym}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"race_id": str, "horse_id": str, "jockey_id": str, "trainer_id": str,
                                 "owner_id": str, "combo": str})


def write_month(kind: str, ym: str, df: pd.DataFrame) -> None:
    p = RAW / kind / f"{ym}.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})


def fetch_text(f: Fetcher, url: str, referer: str | None = None) -> str:
    status, body = f.get_bytes(url, referer=referer)
    if status in (400, 403, 429):
        raise Blocked(f"HTTP {status} for {url}")
    if status != 200:
        return ""
    return decode(body)


def kaisai_dates(f: Fetcher, ym: str) -> list[str]:
    y, m = int(ym[:4]), int(ym[4:])
    dates = nk.parse_calendar(fetch_text(f, nk.url_calendar(y, m)))
    dates = [d for d in dates if d.startswith(ym)]
    if dates:
        return dates
    # Fallback when the calendar has nothing: weekends and Mondays (holiday meetings).
    d = dt.date(y, m, 1)
    out = []
    while d.month == m:
        if d.weekday() in (5, 6, 0):
            out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


def scrape_month(f: Fetcher, ym: str, deadline: float) -> tuple[bool, int]:
    """Scrape one month. Returns (complete, n_new_races)."""
    today = today_jst().strftime("%Y%m%d")
    races, results, payouts = read_month("races", ym), read_month("results", ym), read_month("payouts", ym)
    have = set(races["race_id"]) if len(races) else set()
    new_r, new_x, new_p = [], [], []
    complete = True
    try:
        for date in kaisai_dates(f, ym):
            if date > today:
                complete = False
                continue
            ids = nk.parse_db_race_list(fetch_text(f, nk.url_db_race_list(date)))
            for rid in ids:
                if rid in have:
                    continue
                if time.monotonic() > deadline:
                    complete = False
                    break
                parsed = nk.parse_db_race(fetch_text(f, nk.url_db_race(rid)), rid)
                if parsed is None:  # not run yet / abandoned / page not published
                    if date >= (today_jst() - dt.timedelta(days=10)).strftime("%Y%m%d"):
                        complete = False
                    continue
                new_r.append(parsed["race"])
                new_x.extend(parsed["runners"])
                new_p.extend(parsed["payouts"])
                have.add(rid)
            if time.monotonic() > deadline:
                complete = False
                break
    finally:
        if new_r:
            write_month("races", ym, pd.concat([races, pd.DataFrame(new_r)], ignore_index=True)
                        .sort_values("race_id"))
            write_month("results", ym, pd.concat([results, pd.DataFrame(new_x)], ignore_index=True)
                        .sort_values(["race_id", "number"]))
            write_month("payouts", ym, pd.concat([payouts, pd.DataFrame(new_p)], ignore_index=True)
                        .sort_values(["race_id", "bet_type"], kind="stable"))
    if ym >= today[:6]:
        complete = False  # current month keeps growing
    return complete, len(new_r)


def git_push(message: str) -> None:
    """Commit data files and push to the current branch (used inside GitHub Actions)."""
    def run(*a: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(a, cwd=ROOT.parent, check=check, capture_output=True, text=True)

    run("git", "add", "jra_keiba/data/raw")
    if run("git", "diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    run("git", "commit", "-q", "-m", message)
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    for attempt in range(4):
        run("git", "pull", "-q", "--rebase", "origin", branch, check=False)
        if run("git", "push", "-q", "origin", f"HEAD:{branch}", check=False).returncode == 0:
            print(f"pushed: {message}", flush=True)
            return
        time.sleep(2 ** (attempt + 1))
    print("WARN: push failed", flush=True)


def cmd_results(args) -> int:
    f = Fetcher(min_interval=args.interval)
    start = time.monotonic()
    deadline = start + args.max_minutes * 60
    state = load_state()
    last_push = time.monotonic()
    pending: list[str] = []
    rc = 0
    for ym in months_desc(args.start, args.end):
        if ym in state["complete_months"]:
            continue
        if time.monotonic() > deadline:
            break
        try:
            complete, n = scrape_month(f, ym, deadline)
        except Blocked as exc:
            print(f"BLOCKED: {exc}", flush=True)
            rc = 2
            break
        print(f"{ym}: +{n} races complete={complete} requests={f.n_requests} "
              f"elapsed={(time.monotonic() - start) / 60:.1f}min", flush=True)
        if complete:
            state["complete_months"].append(ym)
        save_state(state)
        if n:
            pending.append(ym)
        if args.push and pending and time.monotonic() - last_push > args.push_every * 60:
            git_push(f"Add JRA race results {pending[-1]}..{pending[0]}")
            pending, last_push = [], time.monotonic()
    save_state(state)
    if args.push:
        git_push(f"Add JRA race results {pending[-1]}..{pending[0]}" if pending else "Update scrape state")
    return rc


def cmd_odds(args) -> int:
    """Final odds of every ticket (all bet types) for graded races, for EV backtests."""
    f = Fetcher(min_interval=args.interval)
    deadline = time.monotonic() + args.max_minutes * 60
    grades = set(args.grades.split(","))
    state = load_state()
    done = set(state.get("odds_done", []))
    last_push = time.monotonic()
    rc = 0
    for ym in months_desc(args.start, args.end):
        races = read_month("races", ym)
        if not len(races):
            continue
        target = [r for r in races.loc[races["grade"].isin(grades), "race_id"] if r not in done]
        if not target:
            continue
        rows = []
        try:
            for rid in target:
                if time.monotonic() > deadline:
                    break
                ok = True
                for t in (1, 2, 4, 5, 6, 7, 8):
                    if t == 2:
                        continue  # type 1 response already carries 複勝
                    body = fetch_text(f, nk.url_odds(rid, t), referer=nk.url_shutuba(rid))
                    types = (1, 2) if t == 1 else (t,)
                    for tt in types:
                        o = nk.parse_odds(body, tt)
                        if o["status"] != "result" or not o["odds"]:
                            ok = ok and tt == 2
                            continue
                        for combo, (lo, hi) in o["odds"].items():
                            rows.append((rid, nk.ODDS_TYPE[tt], "-".join(map(str, combo)), lo, hi))
                    if not ok:
                        break
                done.add(rid)  # also marks races whose odds are no longer served
                print(f"odds {rid} ok={ok} requests={f.n_requests}", flush=True)
        except Blocked as exc:
            print(f"BLOCKED: {exc}", flush=True)
            rc = 2
        if rows:
            old = read_month("odds", ym)
            new = pd.DataFrame(rows, columns=["race_id", "bet_type", "combo", "odds", "odds_max"])
            write_month("odds", ym, pd.concat([old, new], ignore_index=True))
        state["odds_done"] = sorted(done)
        save_state(state)
        if args.push and time.monotonic() - last_push > args.push_every * 60:
            git_push(f"Add final odds for graded races {ym}")
            last_push = time.monotonic()
        if rc or time.monotonic() > deadline:
            break
    if args.push:
        git_push("Add final odds for graded races")
    return rc


def cmd_pedigree(args) -> int:
    """Sire / dam / broodmare sire. One past-performance card covers a whole field, so
    races are chosen greedily by how many still-unknown horses they contain."""
    f = Fetcher(min_interval=args.interval)
    deadline = time.monotonic() + args.max_minutes * 60
    ped_path = RAW / "pedigree.csv.gz"
    ped = pd.read_csv(ped_path, dtype=str) if ped_path.exists() else pd.DataFrame(
        columns=["horse_id", "sire", "dam", "bms"])
    known = set(ped["horse_id"])
    state = load_state()
    tried = set(state.get("ped_races_done", []))
    frames = [read_month("results", p.name[:6])[["race_id", "horse_id"]]
              for p in sorted((RAW / "results").glob("*.csv.gz"), reverse=True)]
    if not frames:
        return 0
    res = pd.concat(frames, ignore_index=True)
    res = res[res["race_id"].str[:4] >= str(args.since)]
    rows, rc, n = [], 0, 0
    last_push = time.monotonic()

    def flush() -> None:
        nonlocal ped, rows
        if rows:
            ped = pd.concat([ped, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("horse_id")
            ped.sort_values("horse_id").to_csv(ped_path, index=False, compression={"method": "gzip", "mtime": 0})
            rows = []
        state["ped_races_done"] = sorted(tried)
        save_state(state)

    try:
        # newest races first: one pass, skipping races whose horses are all known by then
        for rid, grp in res.groupby("race_id", sort=False):
            if time.monotonic() > deadline:
                break
            if rid in tried:
                continue
            unknown = set(grp["horse_id"]) - known
            if len(unknown) < args.min_unknown:
                continue
            got = nk.parse_shutuba_past_pedigree(fetch_text(f, nk.url_shutuba_past(rid)))
            tried.add(rid)
            for g in got:
                if g["horse_id"] not in known:
                    known.add(g["horse_id"])
                    rows.append(g)
            n += 1
            if n % 200 == 0:
                print(f"pedigree: {n} cards, {len(known)} horses known, requests={f.n_requests}", flush=True)
                flush()
                if args.push and time.monotonic() - last_push > args.push_every * 60:
                    git_push("Add pedigree data")
                    last_push = time.monotonic()
    except Blocked as exc:
        print(f"BLOCKED: {exc}", flush=True)
        rc = 2
    flush()
    print(f"pedigree: done {n} cards, {len(known)} horses known", flush=True)
    if args.push:
        git_push("Add pedigree data")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    this_month = today_jst().strftime("%Y-%m")
    for name in ("results", "odds", "pedigree"):
        p = sub.add_parser(name)
        p.add_argument("--from", dest="start", default=this_month, help="newest month YYYY-MM")
        p.add_argument("--to", dest="end", default=this_month, help="oldest month YYYY-MM")
        p.add_argument("--max-minutes", type=float, default=330)
        p.add_argument("--interval", type=float, default=3.0, help="minimum seconds between requests")
        p.add_argument("--push", action="store_true", help="git commit+push data periodically")
        p.add_argument("--push-every", type=float, default=40, help="minutes between pushes")
        if name == "odds":
            p.add_argument("--grades", default="G1,G2,G3")
        if name == "pedigree":
            p.add_argument("--since", type=int, default=2000, help="only races from this year on")
            p.add_argument("--min-unknown", type=int, default=1)
    args = ap.parse_args(argv)
    return {"results": cmd_results, "odds": cmd_odds, "pedigree": cmd_pedigree}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
