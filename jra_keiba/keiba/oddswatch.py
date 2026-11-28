"""Record the 単勝 odds of every JRA race of a day shortly before post time.

Backtests settle at the final odds, but a bet is placed some minutes earlier, and the odds
keep moving until the close. This keeps a snapshot of the whole board N minutes before
each race (one request per race), so the movement to the final odds can be measured on
every race instead of guessed: data/raw/odds_snapshots/YYYYMMDD.csv.gz.

Usage (inside GitHub Actions; the job sleeps between races):
    python -m keiba.oddswatch --date 20261121 --lead 15 --from-time 09:00 --to-time 13:00 --push
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from . import netkeiba as nk
from .http import Fetcher
from .scrape import JST, RAW, Blocked, fetch_text, git_push, today_jst

OUT = RAW / "odds_snapshots"
COLS = ["race_id", "start_time", "number", "odds", "odds_time", "fetched_at", "minutes_to_post", "status", "lead"]


def snapshot(interval: float, race_id: str, post: dt.datetime) -> list[dict]:
    """The board of one race. A new HTTP session is opened for every race: on the first day
    the watcher ran, every request made more than about an hour after its session started
    came back without odds. An empty answer is logged and tried once more with the other
    API action."""
    from .http import decode
    base = nk.url_odds(race_id, 1)
    o = {"odds": {}, "time": None, "status": None}
    for action in ("update", "init"):
        f = Fetcher(min_interval=interval)
        status, body = f.get_bytes(base.replace("action=init", f"action={action}"), referer=nk.url_shutuba(race_id))
        if status in (400, 403, 429):
            raise Blocked(f"HTTP {status} for odds of {race_id}")
        text = decode(body) if status == 200 else ""
        o = nk.parse_odds(text, 1)
        if o["odds"]:
            break
        print(f"  {race_id}: no odds with action={action} (HTTP {status}, status={o['status']}, "
              f"body starts {text[:160]!r})", flush=True)
    now = dt.datetime.now(JST)
    return [{"race_id": race_id, "start_time": post.strftime("%H:%M"), "number": k[0], "odds": v[0],
             "odds_time": o["time"], "fetched_at": now.isoformat(timespec="seconds"),
             "minutes_to_post": round((post - now).total_seconds() / 60.0, 1), "status": o["status"]}
            for k, v in o["odds"].items()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="", help="YYYYMMDD (default: today in Japan)")
    ap.add_argument("--lead", default="15", help="minutes before post time; several as 30,15,10,5")
    ap.add_argument("--from-time", default="00:00", help="only races starting at or after HH:MM")
    ap.add_argument("--to-time", default="23:59", help="only races starting before HH:MM")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args(argv)
    leads = sorted({float(x) for x in str(args.lead).split(",") if x.strip()}, reverse=True)
    date = args.date or today_jst().strftime("%Y%m%d")
    f = Fetcher(min_interval=args.interval)
    day = dt.datetime.strptime(date, "%Y%m%d").replace(tzinfo=JST)
    path = OUT / f"{date}.csv.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(path, dtype={"race_id": str}).to_dict("records") if path.exists() else []
    for r in rows:                              # files written before several leads were recorded
        if pd.isna(r.get("lead")):
            r["lead"] = 15.0
    done = {(r["race_id"], float(r["lead"])) for r in rows}
    last_push = time.monotonic()

    def save(push: bool) -> None:
        nonlocal last_push
        if rows:
            pd.DataFrame(rows, columns=COLS).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        if push and args.push:
            git_push(f"Add pre-race odds snapshots {date}")
            last_push = time.monotonic()

    try:
        events = []
        for r in nk.parse_race_list_sub(fetch_text(f, nk.url_race_list_sub(date))):
            try:
                h, m = map(int, r["start_time"].split(":"))
            except (ValueError, AttributeError):
                continue
            if not args.from_time <= f"{h:02d}:{m:02d}" < args.to_time:
                continue
            post = day.replace(hour=h, minute=m)
            events += [(post - dt.timedelta(minutes=lead), post, r["race_id"], lead) for lead in leads
                       if (r["race_id"], lead) not in done]
        print(f"{date}: {len(events)} snapshots to take ({len({e[2] for e in events})} races, leads {leads})", flush=True)
        for when, post, rid, lead in sorted(events):
            now = dt.datetime.now(JST)
            if now > when + dt.timedelta(minutes=2) or now > post - dt.timedelta(minutes=1):
                print(f"skip {rid} at {lead:g} min: too late", flush=True)     # missed its moment
                continue
            wait = (when - now).total_seconds()
            if wait > 0:
                time.sleep(wait)
            got = snapshot(args.interval, rid, post)
            for g in got:
                g["lead"] = lead
            rows += got
            print(f"{rid} post {post:%H:%M} at {lead:g} min: {len(got)} runners", flush=True)
            save(push=time.monotonic() - last_push > 1200)     # at most 20 minutes of snapshots at risk
    except Blocked as exc:
        print(f"BLOCKED: {exc}", flush=True)
        save(push=True)
        return 2
    save(push=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
