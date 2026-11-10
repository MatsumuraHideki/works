"""Fetch a handful of pages and save the raw bytes, to inspect real page structure."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .http import Fetcher, decode

SEED = {
    "db_race_2025.html": "https://db.netkeiba.com/race/202505021211/",
    "db_race_2015.html": "https://db.netkeiba.com/race/201505021211/",
    "db_race_2005.html": "https://db.netkeiba.com/race/200505021210/",
    "db_race_1995.html": "https://db.netkeiba.com/race/199505030610/",
    "db_racelist_20251102.html": "https://db.netkeiba.com/race/list/20251102/",
    "db_racelist_20261108.html": "https://db.netkeiba.com/race/list/20261108/",
    "db_horse_ped.html": "https://db.netkeiba.com/horse/ped/2022105081/",
    "db_horse.html": "https://db.netkeiba.com/horse/2022105081/",
    "db_horse_result.html": "https://db.netkeiba.com/horse/result/2022105081/",
    "race_calendar_202611.html": "https://race.netkeiba.com/top/calendar.html?year=2026&month=11",
    "race_calendar_201503.html": "https://race.netkeiba.com/top/calendar.html?year=2015&month=3",
    "race_list_sub_20261115.html": "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20261115",
    "race_list_sub_20261114.html": "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20261114",
    "race_list_sub_20261108.html": "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20261108",
    "race_grade_list.html": "https://race.netkeiba.com/top/schedule.html?year=2026",
    "jra_top.html": "https://www.jra.go.jp/keiba/",
}


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    f = Fetcher(min_interval=1.5)
    texts: dict[str, str] = {}
    for name, url in SEED.items():
        try:
            status, body = f.get_bytes(url)
        except Exception as exc:  # noqa: BLE001
            print(f"ERR  {url} {exc!r}")
            continue
        print(f"{status} {len(body):>8} {url}")
        (out / name).write_bytes(body)
        texts[name] = decode(body)

    # Follow up: upcoming and just-finished races discovered from the race lists.
    def ids(name: str) -> list[str]:
        return sorted(set(re.findall(r"race_id=(\d{12})", texts.get(name, ""))))

    follow: dict[str, str] = {}
    upcoming = ids("race_list_sub_20261115.html") + ids("race_list_sub_20261114.html")
    done = ids("race_list_sub_20261108.html")
    print("upcoming ids", upcoming)
    print("done ids", done)
    for rid in [r for r in upcoming if r.endswith("11")][:4]:
        follow[f"shutuba_{rid}.html"] = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        follow[f"shutuba_past_{rid}.html"] = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={rid}"
        for t in (1, 4):
            follow[f"odds_{rid}_t{t}.json"] = (
                f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type={t}&action=init"
            )
    for rid in [r for r in done if r.endswith("11")][:2]:
        follow[f"result_{rid}.html"] = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
        follow[f"shutuba_done_{rid}.html"] = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        follow[f"db_race_{rid}.html"] = f"https://db.netkeiba.com/race/{rid}/"
        for t in (1, 2, 3, 4, 5, 6, 7, 8):
            follow[f"odds_done_{rid}_t{t}.json"] = (
                f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type={t}&action=init"
            )
    for name, url in follow.items():
        try:
            status, body = f.get_bytes(url, referer="https://race.netkeiba.com/")
        except Exception as exc:  # noqa: BLE001
            print(f"ERR  {url} {exc!r}")
            continue
        print(f"{status} {len(body):>8} {url}")
        (out / name).write_bytes(body)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
