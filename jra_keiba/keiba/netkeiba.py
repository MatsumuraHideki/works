"""URL builders and HTML/JSON parsers for netkeiba pages (JRA races only)."""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup

PLACE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
PLACE_BY_NAME = {v: k for k, v in PLACE.items()}

# race.netkeiba.com grade icons -> label
GRADE_ICON = {"1": "G1", "2": "G2", "3": "G3", "4": "重賞", "5": "OP", "10": "JG1", "11": "JG2",
              "12": "JG3", "15": "L", "16": "3勝", "17": "2勝", "18": "1勝"}

# odds API "type" parameter -> bet type
ODDS_TYPE = {1: "単勝", 2: "複勝", 3: "枠連", 4: "馬連", 5: "ワイド", 6: "馬単", 7: "三連複", 8: "三連単"}


def url_calendar(year: int, month: int) -> str:
    return f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"


def url_db_race_list(date: str) -> str:
    return f"https://db.netkeiba.com/race/list/{date}/"


def url_db_race(race_id: str) -> str:
    return f"https://db.netkeiba.com/race/{race_id}/"


def url_race_list_sub(date: str) -> str:
    return f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"


def url_shutuba(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def url_shutuba_past(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"


def url_odds(race_id: str, type_: int) -> str:
    return f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={type_}&action=init"


def is_jra(race_id: str) -> bool:
    return len(race_id) == 12 and race_id[4:6] in PLACE


def _txt(el) -> str:
    return unicodedata.normalize("NFKC", el.get_text(" ", strip=True)) if el is not None else ""


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _id_from(el, pattern: str) -> str | None:
    if el is None:
        return None
    for a in el.find_all("a"):
        m = re.search(pattern, a.get("href", ""))
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------- calendars / lists

def parse_calendar(html: str) -> list[str]:
    return sorted(set(re.findall(r"kaisai_date=(\d{8})", html)))


def parse_db_race_list(html: str) -> list[str]:
    return sorted(r for r in set(re.findall(r"/race/(\d{12})/", html)) if is_jra(r))


def parse_race_list_sub(html: str) -> list[dict[str, Any]]:
    """Races of one day from race.netkeiba.com (works before and after the races are run)."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for li in soup.select("li.RaceList_DataItem"):
        a = li.find("a")
        m = re.search(r"race_id=(\d{12})", a.get("href", "")) if a else None
        if not m or not is_jra(m.group(1)):
            continue
        grade = None
        for sp in li.select(".RaceList_ItemTitle span[class*=Icon_GradeType]"):
            for c in sp.get("class", []):
                g = re.fullmatch(r"Icon_GradeType(\d+)", c)
                if g and g.group(1) in GRADE_ICON:
                    grade = GRADE_ICON[g.group(1)]
        out.append({
            "race_id": m.group(1),
            "name": _txt(li.select_one(".ItemTitle")),
            "grade": grade,
            "start_time": _txt(li.select_one(".RaceList_Itemtime")),
            "course": _txt(li.select_one(".RaceList_ItemLong")),
            "is_result": "result.html" in a.get("href", ""),
        })
    return out


# ---------------------------------------------------------------- db race result page

def _parse_time(s: str) -> float | None:
    m = re.fullmatch(r"(?:(\d+):)?(\d+)\.(\d)", s.strip())
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 10


def parse_grade(name: str) -> str | None:
    """Grade from a db.netkeiba race title such as '東京優駿(GI)' / '京都大賞典(GII)'."""
    n = unicodedata.normalize("NFKC", name)
    m = re.search(r"\((J\.?)?(G|Jpn)\s?(III|II|I|[123])\)", n)
    if m:
        lvl = {"I": "1", "II": "2", "III": "3"}.get(m.group(3), m.group(3))
        return ("J" if m.group(1) else "") + "G" + lvl
    if re.search(r"\(L\)", n):
        return "L"
    if re.search(r"\(重賞\)", n):
        return "重賞"
    if re.search(r"\(OP\)", n):
        return "OP"
    return None


def parse_class(cond: str, name: str = "") -> str:
    """Class ladder label from the condition text."""
    t = unicodedata.normalize("NFKC", cond + " " + name)
    if "新馬" in t:
        return "新馬"
    if "未勝利" in t:
        return "未勝利"
    if "未出走" in t:
        return "未勝利"
    if re.search(r"3勝|1600万|1500万", t):
        return "3勝"
    if re.search(r"2勝|1000万|900万|800万", t):
        return "2勝"
    if re.search(r"1勝|500万|400万", t):
        return "1勝"
    if re.search(r"オープン|OP", t):
        return "OP"
    return "不明"


def parse_db_race(html: str, race_id: str) -> dict[str, Any] | None:
    """Parse a db.netkeiba.com race page. Returns None when the race has no result table."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.race_table_01")
    intro = soup.select_one("div.data_intro")
    if table is None or intro is None:
        return None

    name = _txt(intro.find("h1"))
    span = _txt(intro.select_one("dl.racedata span") or intro.find("span"))
    small = _txt(intro.select_one("p.smalltxt"))

    race: dict[str, Any] = {"race_id": race_id, "name": name, "grade": parse_grade(name)}
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small)
    race["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None
    m = re.search(r"(\d+)回(\S+?)(\d+)日目\s*(.*)", small)
    race["kai"] = int(m.group(1)) if m else None
    race["place"] = m.group(2) if m else PLACE.get(race_id[4:6])
    race["day"] = int(m.group(3)) if m else None
    cond = m.group(4).strip() if m else ""
    race["cond"] = cond
    race["class"] = parse_class(cond.split("(")[0], name)
    race["race_no"] = int(race_id[10:12])

    parts = [p.strip() for p in span.split("/")]
    course = parts[0] if parts else ""
    race["course_raw"] = course
    race["surface"] = "障" if course.startswith("障") else ("ダ" if course.startswith("ダ") else ("芝" if course.startswith("芝") else None))
    race["direction"] = "直" if "直線" in course else ("右" if "右" in course else ("左" if "左" in course else None))
    race["inout"] = "外" if "外" in course and "内" not in course else ("内" if "内" in course and "外" not in course else None)
    dm = re.search(r"(\d{3,5})\s*m", course)
    race["distance"] = int(dm.group(1)) if dm else None
    race["weather"] = None
    race["going"] = None
    race["start_time"] = None
    for p in parts[1:]:
        if p.startswith("天候"):
            race["weather"] = p.split(":")[-1].strip() or None
        elif p.startswith("発走"):
            tm = re.search(r"(\d{1,2}:\d{2})", p)
            race["start_time"] = tm.group(1) if tm else None
        elif ":" in p:
            # '芝 : 良' / 'ダート : 稍重' / jumps: '芝 : 良 ダート : 稍重' -> first value
            gm = re.search(r":\s*(良|稍重|稍|重|不良)", p)
            race["going"] = gm.group(1) if gm else None
    if race["going"] == "稍":
        race["going"] = "稍重"

    # result rows, addressed by header name because premium columns vary
    rows = table.find_all("tr")
    header = [re.sub(r"\s+", "", _txt(c)) for c in rows[0].find_all(["th", "td"])]

    def col(prefix: str) -> int | None:
        for i, h in enumerate(header):
            if h.startswith(prefix):
                return i
        return None

    ix = {k: col(k) for k in ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差",
                              "通過", "上り", "単勝", "人気", "馬体重", "調教師", "馬主", "賞金"]}
    runners = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue

        def cell(key: str):
            i = ix.get(key)
            return tds[i] if i is not None and i < len(tds) else None

        def text(key: str) -> str:
            return _txt(cell(key))

        fin_raw = text("着順")
        fm = re.match(r"(\d+)", fin_raw)
        sexage = text("性齢")
        bw = re.match(r"(\d+)\s*\(\s*([+-]?\d+)\s*\)", text("馬体重"))
        bw_only = re.match(r"(\d+)", text("馬体重"))
        trainer = text("調教師")
        region = re.search(r"\[(.)\]", trainer)
        runners.append({
            "race_id": race_id,
            "finish": int(fm.group(1)) if fm else None,
            "finish_raw": fin_raw,
            "bracket": _num(text("枠番")),
            "number": _num(text("馬番")),
            "horse_id": _id_from(cell("馬名"), r"/horse/(\w+)"),
            "horse_name": text("馬名"),
            "sex": sexage[:1] or None,
            "age": _num(sexage[1:]),
            "weight_carried": _num(text("斤量")),
            "jockey_id": _id_from(cell("騎手"), r"/jockey/(?:result/recent/)?(\w+)"),
            "jockey_name": text("騎手"),
            "time_sec": _parse_time(text("タイム")),
            "margin": text("着差") or None,
            "passing": text("通過") or None,
            "last3f": _num(text("上り")),
            "win_odds": _num(text("単勝")),
            "popularity": _num(text("人気")),
            "body_weight": float(bw.group(1)) if bw else (float(bw_only.group(1)) if bw_only else None),
            "body_weight_diff": float(bw.group(2)) if bw else None,
            "trainer_id": _id_from(cell("調教師"), r"/trainer/(?:result/recent/)?(\w+)"),
            "trainer_name": re.sub(r"\[.\]\s*", "", trainer) or None,
            "trainer_region": region.group(1) if region else None,
            "owner_id": _id_from(cell("馬主"), r"/owner/(?:result/recent/)?(\w+)"),
            "prize": _num(text("賞金")) or 0.0,
        })
    if not runners:
        return None
    race["n_runners"] = len(runners)

    payouts = []
    for pt in soup.select("table.pay_table_01"):
        for tr in pt.find_all("tr"):
            th, tds = tr.find("th"), tr.find_all("td")
            if th is None or len(tds) < 2:
                continue
            combos = [unicodedata.normalize("NFKC", s).strip() for s in tds[0].stripped_strings]
            pays = [s for s in tds[1].stripped_strings]
            pops = [s for s in tds[2].stripped_strings] if len(tds) > 2 else []
            for i, c in enumerate(combos):
                combo = re.sub(r"\s*(→|-|－|ー)\s*", "-", c)
                payouts.append({
                    "race_id": race_id,
                    "bet_type": _txt(th),
                    "combo": combo,
                    "payout": _num(pays[i]) if i < len(pays) else None,
                    "popularity": _num(pops[i]) if i < len(pops) else None,
                })

    race["laps"] = None
    race["pace"] = None
    lap = soup.find("table", attrs={"summary": "ラップタイム"})
    if lap is not None:
        for tr in lap.find_all("tr"):
            k, v = _txt(tr.find("th")), _txt(tr.find("td"))
            if k.startswith("ラップ"):
                race["laps"] = v.replace(" ", "")
            elif k.startswith("ペース"):
                race["pace"] = v.replace(" ", "")
    return {"race": race, "runners": runners, "payouts": payouts}


# ---------------------------------------------------------------- race card (upcoming)

def parse_shutuba(html: str, race_id: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.Shutuba_Table")
    if table is None:
        return None
    name_el = soup.select_one(".RaceName")
    grade = None
    if name_el is not None:
        for sp in name_el.select("span[class*=Icon_GradeType]"):
            for c in sp.get("class", []):
                g = re.fullmatch(r"Icon_GradeType(\d+)", c)
                if g and g.group(1) in GRADE_ICON:
                    grade = GRADE_ICON[g.group(1)]
    d1 = _txt(soup.select_one(".RaceData01"))
    d2 = _txt(soup.select_one(".RaceData02"))
    title = _txt(soup.title)
    race: dict[str, Any] = {"race_id": race_id, "name": _txt(name_el), "grade": grade,
                            "race_no": int(race_id[10:12]), "place": PLACE.get(race_id[4:6])}
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
    race["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None
    m = re.search(r"(\d{1,2}:\d{2})", d1)
    race["start_time"] = m.group(1) if m else None
    m = re.search(r"(芝|ダ|障)\D{0,3}?(\d{3,5})m", d1)
    race["surface"] = m.group(1) if m else None
    race["distance"] = int(m.group(2)) if m else None
    after = d1.split("m", 1)[1].split("/")[0] if "m" in d1 else ""
    race["direction"] = "直" if "直" in after else ("右" if "右" in after else ("左" if "左" in after else None))
    race["inout"] = "外" if "外" in after and "内" not in after else ("内" if "内" in after and "外" not in after else None)
    m = re.search(r"天候\s*:\s*(\S+)", d1)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"馬場\s*:\s*(\S+)", d1) or re.search(r"/\s*(良|稍重|稍|重|不良)\s*(?:/|$)", d1)
    race["going"] = {"稍": "稍重"}.get(m.group(1), m.group(1)) if m else None
    m = re.search(r"(\d+)回\s*(\S+)\s*(\d+)日目\s*(.*?)\s*(\d+)頭", d2)
    race["kai"] = int(m.group(1)) if m else None
    race["day"] = int(m.group(3)) if m else None
    race["cond"] = m.group(4) if m else d2
    race["class"] = "OP" if grade in ("G1", "G2", "G3", "L", "OP", "重賞") else parse_class(race["cond"], race["name"])
    race["course_raw"] = d1

    runners = []
    for tr in table.select("tr.HorseList"):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        info = tr.select_one("td.HorseInfo")
        sexage = _txt(tr.select_one("td.Barei"))
        wtd = tr.select_one("td.Weight")
        wt = _txt(wtd)
        bw = re.match(r"(\d+)\s*\(\s*([+-]?\d+)\s*\)", wt)
        bw_only = re.match(r"(\d+)", wt)
        trainer_td = tr.select_one("td.Trainer")
        region = _txt(trainer_td.select_one("span")) if trainer_td else ""
        jockey_td = tr.select_one("td.Jockey")
        cancelled = "Cancel" in " ".join(tr.get("class", []))
        runners.append({
            "race_id": race_id,
            "row_id": (tr.get("id") or "").replace("tr_", ""),
            "bracket": _num(_txt(tds[0])),
            "number": _num(_txt(tds[1])),
            "horse_id": _id_from(info, r"/horse/(\w+)"),
            "horse_name": _txt(info.select_one(".HorseName") if info else None),
            "sex": sexage[:1] or None,
            "age": _num(re.sub(r"\D", "", sexage)),
            "weight_carried": _num(_txt(tds[5])),
            "jockey_id": _id_from(jockey_td, r"/jockey/(?:result/recent/)?(\w+)"),
            "jockey_name": _txt(jockey_td),
            "trainer_id": _id_from(trainer_td, r"/trainer/(?:result/recent/)?(\w+)"),
            "trainer_name": _txt(trainer_td.find("a")) if trainer_td and trainer_td.find("a") else None,
            "trainer_region": {"美浦": "東", "栗東": "西"}.get(region, region[:1] or None),
            "body_weight": float(bw.group(1)) if bw else (float(bw_only.group(1)) if bw_only else None),
            "body_weight_diff": float(bw.group(2)) if bw else None,
            "cancelled": cancelled,
        })
    race["n_runners"] = len([r for r in runners if not r["cancelled"]])
    return {"race": race, "runners": runners}


def parse_shutuba_past_pedigree(html: str) -> list[dict[str, Any]]:
    """Sire / dam / broodmare sire of every runner, from the 5-run past performance card."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("table.Shutuba_Past5_Table tr.HorseList"):
        info = tr.select_one("td.Horse_Info")
        if info is None:
            continue
        hid = _id_from(info.select_one(".Horse02"), r"/horse/(\w+)")
        if not hid:
            continue
        bms = _txt(info.select_one(".Horse04")).strip("() ")
        out.append({
            "horse_id": hid,
            "sire": _txt(info.select_one(".Horse01")) or None,
            "dam": _txt(info.select_one(".Horse03")) or None,
            "bms": bms or None,
        })
    return out


# ---------------------------------------------------------------- odds API

def parse_odds(body: str, type_: int) -> dict[str, Any]:
    """Return {'status', 'time', 'odds': {combo_tuple: (low, high)}} for one bet type.

    status: 'result' (final), 'middle' (live), 'yoso' (predicted, keyed by card row id), 'NG' (none).
    For 複勝/ワイド the API gives a (min, max) range; other types have max == min.
    """
    try:
        js = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "NG", "time": None, "odds": {}}
    data = js.get("data") or {}
    table = (data.get("odds") or {}).get(str(type_), {}) if isinstance(data, dict) else {}
    out: dict[tuple[int, ...], tuple[float, float]] = {}
    for key, val in table.items():
        lo = _num(val[0]) if len(val) > 0 else None
        hi = _num(val[1]) if len(val) > 1 else None
        if lo is None or lo <= 0:
            continue
        if not hi or hi <= 0:
            hi = lo
        width = 2 if len(key) > 2 else len(key)
        combo = tuple(int(key[i:i + width]) for i in range(0, len(key), width))
        out[combo] = (lo, hi)
    return {"status": js.get("status"), "time": data.get("official_datetime") if isinstance(data, dict) else None,
            "odds": out}
