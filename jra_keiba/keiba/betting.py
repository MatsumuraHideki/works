"""Turn win probabilities into probabilities, expected values and stakes for every JRA bet type.

Finishing-order model: discounted Harville (Plackett-Luce with a separate
exponent for 2nd and 3rd place). Plain Harville overstates how often favourites
run 2nd/3rd; the exponents ``lam2`` / ``lam3`` (<1) flatten the conditional
distributions and are fitted on historical data (see model.fit_place_discount).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np

BET_TYPES = ("単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単")
# JRA payout rates (払戻率); 1 - rate is the takeout.
PAYOUT_RATE = {
    "単勝": 0.80, "複勝": 0.80, "枠連": 0.775, "馬連": 0.775, "ワイド": 0.775,
    "馬単": 0.75, "三連複": 0.75, "三連単": 0.725,
}


def order_probs(p: np.ndarray, lam2: float = 0.85, lam3: float = 0.75,
                a: np.ndarray | None = None, b: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (P2, P3): P2[i,j]=P(i 1st, j 2nd), P3[i,j,k]=P(i 1st, j 2nd, k 3rd).

    Second and third place are drawn in proportion to strengths ``a`` and ``b`` among the
    runners still unplaced; by default these are p ** lam2 and p ** lam3."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(p)
    a = p ** lam2 if a is None else np.asarray(a, dtype=float)
    b = p ** lam3 if b is None else np.asarray(b, dtype=float)
    eye = np.eye(n, dtype=bool)
    # second given first
    denom2 = a.sum() - a  # by first horse i
    c2 = np.where(eye, 0.0, a[None, :] / denom2[:, None])
    P2 = p[:, None] * c2
    if n < 3:
        return P2, np.zeros((n, n, n))
    denom3 = b.sum() - b[:, None] - b[None, :]  # by (i, j)
    with np.errstate(divide="ignore", invalid="ignore"):
        c3 = b[None, None, :] / denom3[:, :, None]
    mask = eye[:, :, None] | eye[:, None, :] | eye[None, :, :]
    c3 = np.where(mask, 0.0, c3)
    P3 = P2[:, :, None] * c3
    return P2, P3


def place_slots(n_runners: int) -> int:
    """Number of paying 複勝 places under JRA rules (by number of starters)."""
    if n_runners >= 8:
        return 3
    if n_runners >= 5:
        return 2
    return 0


def bet_probabilities(p: np.ndarray, numbers: list[int], lam2: float = 0.85, lam3: float = 0.75,
                      a: np.ndarray | None = None, b: np.ndarray | None = None
                      ) -> dict[str, dict[tuple[int, ...], float]]:
    """Probability of every ticket of every bet type. Keys are tuples of 馬番.

    Unordered bet types use ascending tuples; ordered types (馬単, 三連単) use
    finishing order.
    """
    n = len(p)
    P2, P3 = order_probs(p, lam2, lam3, a, b)
    idx = range(n)
    out: dict[str, dict[tuple[int, ...], float]] = {t: {} for t in BET_TYPES}
    pn = np.asarray(p, dtype=float) / np.sum(p)
    for i in idx:
        out["単勝"][(numbers[i],)] = float(pn[i])
    # top-2 / top-3 membership
    second = P2.sum(axis=0)
    third = P3.sum(axis=(0, 1)) if n >= 3 else np.zeros(n)
    slots = place_slots(n)
    if slots:
        top = pn + second + (third if slots == 3 else 0.0)
        for i in idx:
            out["複勝"][(numbers[i],)] = float(top[i])
    if n >= 2:
        for i, j in combinations(idx, 2):
            key = tuple(sorted((numbers[i], numbers[j])))
            out["馬連"][key] = float(P2[i, j] + P2[j, i])
        for i, j in permutations(idx, 2):
            out["馬単"][(numbers[i], numbers[j])] = float(P2[i, j])
    if n >= 3:
        # pair both in top 3: sum over orders with a third horse anywhere
        S = P3 + P3.transpose(0, 2, 1) + P3.transpose(1, 0, 2) + P3.transpose(1, 2, 0) \
            + P3.transpose(2, 0, 1) + P3.transpose(2, 1, 0)  # S[i,j,k] = P({i,j,k} top3)
        pair_top3 = S.sum(axis=2)  # each unordered triple counted once per k
        for i, j in combinations(idx, 2):
            key = tuple(sorted((numbers[i], numbers[j])))
            out["ワイド"][key] = float(pair_top3[i, j])
        for i, j, k in combinations(idx, 3):
            key = tuple(sorted((numbers[i], numbers[j], numbers[k])))
            out["三連複"][key] = float(S[i, j, k])
        for i, j, k in permutations(idx, 3):
            out["三連単"][(numbers[i], numbers[j], numbers[k])] = float(P3[i, j, k])
    return out


@dataclass
class Bet:
    bet_type: str
    combo: tuple[int, ...]
    prob: float
    odds: float
    ev: float          # expected return per 1 unit staked (1.0 = break-even)
    kelly: float       # full-Kelly fraction of bankroll
    stake: int = 0     # yen, filled by size_bets

    def label(self) -> str:
        sep = "→" if self.bet_type in ("馬単", "三連単") else "-"
        return sep.join(str(c) for c in self.combo)


def evaluate_bets(probs: dict[str, dict[tuple[int, ...], float]],
                  odds: dict[str, dict[tuple[int, ...], float]]) -> list[Bet]:
    """Join model probabilities with market odds. Odds are decimal (払戻/100)."""
    bets: list[Bet] = []
    for bt, table in odds.items():
        pt = probs.get(bt, {})
        for combo, o in table.items():
            if o is None or not np.isfinite(o) or o <= 1.0:
                continue
            pr = pt.get(combo)
            if pr is None or pr <= 0:
                continue
            ev = pr * o
            kelly = max(0.0, (ev - 1.0) / (o - 1.0))
            bets.append(Bet(bt, combo, pr, float(o), float(ev), float(kelly)))
    return bets


def select_bets(bets: list[Bet], min_ev: float = 1.15, min_prob: float = 0.005,
                max_bets: int = 12, max_per_type: int = 4) -> list[Bet]:
    """Keep positive-EV tickets with a safety margin.

    ``min_ev`` > 1 is a margin for model error; ``min_prob`` drops extreme long
    shots where a tiny absolute probability error produces a huge fake edge.
    Ranking is by expected log-growth contribution (kelly * (ev-1)), which
    favours edges that are both large and likely, rather than raw EV.
    """
    good = [b for b in bets if b.ev >= min_ev and b.prob >= min_prob]
    good.sort(key=lambda b: b.kelly * (b.ev - 1.0), reverse=True)
    picked: list[Bet] = []
    per_type: dict[str, int] = {}
    for b in good:
        if per_type.get(b.bet_type, 0) >= max_per_type:
            continue
        picked.append(b)
        per_type[b.bet_type] = per_type.get(b.bet_type, 0) + 1
        if len(picked) >= max_bets:
            break
    return picked


def size_bets(bets: list[Bet], bankroll: int = 100_000, kelly_fraction: float = 0.25,
              max_race_fraction: float = 0.05, unit: int = 100) -> list[Bet]:
    """Fractional-Kelly stakes, capped per race, rounded down to 100-yen units."""
    raw = np.array([b.kelly * kelly_fraction * bankroll for b in bets], dtype=float)
    cap = bankroll * max_race_fraction
    if raw.sum() > cap and raw.sum() > 0:
        raw *= cap / raw.sum()
    for b, s in zip(bets, raw):
        b.stake = int(s // unit) * unit
    return [b for b in bets if b.stake > 0]


def portfolio_ev(bets: list[Bet]) -> tuple[int, float]:
    """(total stake, expected return) of a sized set of bets."""
    stake = sum(b.stake for b in bets)
    ret = sum(b.stake * b.ev for b in bets)
    return stake, ret


def budget_slip(bets: list[Bet], budget: int, unit: int = 100) -> list[Bet]:
    """Fit tickets into a budget: one unit each in order of hit probability until the money
    runs out; money left after every ticket has a unit is added from the top, a unit at a time."""
    ordered = sorted(bets, key=lambda b: -b.prob)
    n_units = budget // unit
    for b in ordered:
        b.stake = 0
    if not ordered or n_units <= 0:
        return []
    i = 0
    while n_units > 0:
        ordered[i % len(ordered)].stake += unit
        n_units -= 1
        i += 1
    return [b for b in ordered if b.stake > 0]


CORE_TYPES = ("馬連", "ワイド", "三連複")
# Rules of the two published slips. They were chosen in reports/experiments.md: stakes in
# proportion to the Kelly fraction did better than flat stakes in both halves of the test
# period, and ワイド tickets and long ticket lists pulled the mixed slip down.
WIN_EV = 1.00          # 単勝 threshold in the mixed slip
# 単勝 threshold of the win-only slip. Backtests use closing odds; a slip is bought before the
# close, and bets with EV 1.00-1.10 only broke even at closing odds, so the win-only slip
# keeps a margin for the odds moving after purchase.
WIN_SLIP_EV = 1.10
KELLY_FULL = 0.05      # Kelly fraction at which one horse gets the whole race budget
SLIP_TYPES = ("馬連", "三連複")   # multi-horse bet types of the mixed slip
SLIP_TICKETS = 20      # at most this many tickets in the mixed slip


def kelly_fraction(prob: float, odds: float) -> float:
    """Full-Kelly share of the bankroll for one bet (0 when the bet has no edge)."""
    return max(0.0, (prob * odds - 1.0) / (odds - 1.0)) if odds > 1.0 else 0.0


def allocate(weights: list[float], budget: int, unit: int = 100) -> list[int]:
    """Split a budget in proportion to ``weights`` in whole units (largest remainder).
    With more tickets than units, the heaviest get one unit each."""
    n_units = budget // unit
    total = sum(weights)
    out = [0] * len(weights)
    if not weights or total <= 0 or n_units <= 0:
        return out
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    if len(weights) > n_units:
        for i in order[:n_units]:
            out[i] = unit
        return out
    raw = [w / total * n_units for w in weights]
    base = [int(x) for x in raw]
    left = n_units - sum(base)
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - base[i]))[:left]:
        base[i] += 1
    return [b * unit for b in base]


def make_slips(cand: list[Bet], top_number: int, budget: int, min_ev: float = 1.0
               ) -> tuple[list[Bet], list[Bet]]:
    """The two budget slips of a race, from every priced ticket (``cand``).

    A (win only)  単勝 with EV >= WIN_SLIP_EV. Each horse gets the budget times
                  min(1, Kelly fraction / KELLY_FULL): more on the horses the model is surer
                  of, the whole budget only on a strong edge. The rest is not spent.
    B (mixed)     単勝 with EV >= WIN_EV plus 馬連 / 三連複 with EV >= ``min_ev``: the
                  SLIP_TICKETS likeliest, with the budget split in proportion to the Kelly
                  fraction of each ticket.

    Predictions and the backtest both call this, so the reported record is for exactly the
    slips that are published. ``top_number`` is unused (kept for callers).
    """
    def copy(b: Bet) -> Bet:
        return Bet(b.bet_type, b.combo, b.prob, b.odds, b.ev, kelly_fraction(b.prob, b.odds))

    unit = 100
    win_slip = [copy(b) for b in cand if b.bet_type == "単勝" and b.ev >= WIN_SLIP_EV and b.prob >= 0.01]
    for b in win_slip:
        b.stake = max(unit, int(budget * min(1.0, b.kelly / KELLY_FULL) // unit) * unit)
    if sum(b.stake for b in win_slip) > budget:
        for b, st in zip(win_slip, allocate([b.kelly for b in win_slip], budget, unit)):
            b.stake = st
    win_slip = sorted((b for b in win_slip if b.stake > 0), key=lambda b: -b.prob)

    pool = [copy(b) for b in cand if (b.bet_type == "単勝" and b.ev >= WIN_EV and b.prob >= 0.01)
            or (b.bet_type in SLIP_TYPES and b.ev >= min_ev and b.prob >= 0.005)]
    pool = sorted(pool, key=lambda b: -b.prob)[:SLIP_TICKETS]
    for b, st in zip(pool, allocate([max(b.kelly, 1e-9) for b in pool], budget, unit)):
        b.stake = st
    slip = [b for b in pool if b.stake > 0]
    return win_slip, slip


# ------------------------------------------------------------------ grouping for entry

def _max_clique(adj: dict[int, set[int]]) -> list[int]:
    """Largest clique of a small graph (fields have at most 18 runners)."""
    best: list[int] = []

    def grow(r: list[int], p: set[int]) -> None:
        nonlocal best
        if len(r) > len(best):
            best = r[:]
        for v in sorted(p, key=lambda x: -len(adj[x] & p)):
            if len(r) + len(p) <= len(best):
                return
            grow(r + [v], p & adj[v])
            p = p - {v}

    grow([], set(adj))
    return sorted(best)


def group_pairs(pairs: list[tuple[int, int]]) -> list[tuple[str, list[int], list[int]]]:
    """Cover a set of two-horse tickets exactly with boxes, one-axis wheels and singles.

    Returns (kind, axis, others): kind 'ボックス' (others = the boxed horses),
    'ながし' (axis = [a], others = partners) or '通常' (others = the pair).
    Greedy: always take the form that covers the most remaining tickets.
    """
    left = {tuple(sorted(p)) for p in pairs}
    out = []
    while left:
        adj: dict[int, set[int]] = {}
        for a, b in left:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        star = max(adj, key=lambda v: (len(adj[v]), -v))
        clique = _max_clique(adj)
        n_box = len(clique) * (len(clique) - 1) // 2
        if len(clique) >= 3 and n_box > len(adj[star]):
            out.append(("ボックス", [], clique))
            left -= {(a, b) for i, a in enumerate(clique) for b in clique[i + 1:]}
        elif len(adj[star]) >= 2:
            out.append(("ながし", [star], sorted(adj[star])))
            left -= {tuple(sorted((star, b))) for b in adj[star]}
        else:
            p = min(left)
            out.append(("通常", [], list(p)))
            left.discard(p)
    return out


def group_triples(triples: list[tuple[int, int, int]]) -> list[tuple[str, list[int], list[int]]]:
    """Cover a set of 三連複 tickets exactly with boxes, wheels and singles.

    kinds: 'ボックス' (others = horses), '軸1頭ながし' (axis=[a], every pair of others),
    '軸2頭ながし' (axis=[a, b], each of others as the third horse), '通常'.
    """
    left = {tuple(sorted(t)) for t in triples}
    out = []
    while left:
        horses = sorted({h for t in left for h in t})
        best = ("通常", [], list(min(left)), 1)
        # two fixed horses + any third
        pair_cnt: dict[tuple[int, int], list[int]] = {}
        for a, b, c in left:
            pair_cnt.setdefault((a, b), []).append(c)
            pair_cnt.setdefault((a, c), []).append(b)
            pair_cnt.setdefault((b, c), []).append(a)
        pk = max(pair_cnt, key=lambda k: (len(pair_cnt[k]), -k[0], -k[1]))
        if len(pair_cnt[pk]) >= 2:
            best = ("軸2頭ながし", list(pk), sorted(pair_cnt[pk]), len(pair_cnt[pk]))
        # one fixed horse + every pair among a set (a clique in that horse's link graph)
        for a in horses:
            adj: dict[int, set[int]] = {}
            for t in left:
                if a in t:
                    b, c = [x for x in t if x != a]
                    adj.setdefault(b, set()).add(c)
                    adj.setdefault(c, set()).add(b)
            if not adj:
                continue
            cl = _max_clique(adj)
            n = len(cl) * (len(cl) - 1) // 2
            if len(cl) >= 3 and n > best[3]:
                best = ("軸1頭ながし", [a], cl, n)
                # a full box is the same set plus the triples among the partners themselves
                box = sorted([a] + cl)
                inner = {tuple(sorted((x, y, z))) for i, x in enumerate(box) for j, y in enumerate(box[i + 1:], i + 1)
                         for z in box[j + 1:]}
                if inner <= left and len(inner) > n:
                    best = ("ボックス", [], box, len(inner))
        kind, axis, others, _ = best
        if kind == "通常":
            left.discard(tuple(others))
        elif kind == "軸2頭ながし":
            left -= {tuple(sorted(axis + [c])) for c in others}
        elif kind == "軸1頭ながし":
            left -= {tuple(sorted((axis[0], x, y))) for i, x in enumerate(others) for y in others[i + 1:]}
        else:
            left -= {tuple(sorted((x, y, z))) for i, x in enumerate(others) for j, y in enumerate(others[i + 1:], i + 1)
                     for z in others[j + 1:]}
        out.append((kind, axis, others))
    return out


def grouped_slip(bets: list[Bet]) -> list[dict]:
    """A budget slip rewritten as the fewest box / wheel entries, per bet type and stake."""
    out = []
    order = {"単勝": 0, "複勝": 1, "馬連": 2, "ワイド": 3, "三連複": 4}
    keys = sorted({(b.bet_type, b.stake) for b in bets}, key=lambda k: (order.get(k[0], 9), -k[1]))
    for bt, stake in keys:
        combos = [b.combo for b in bets if b.bet_type == bt and b.stake == stake]
        if bt in ("単勝", "複勝"):
            out.append({"bet_type": bt, "kind": "通常", "axis": [], "others": sorted(c[0] for c in combos),
                        "n": len(combos), "stake": stake})
            continue
        groups = group_pairs(combos) if bt in ("馬連", "ワイド") else group_triples(combos)
        for kind, axis, others in groups:
            if kind == "ボックス":
                k = len(others)
                n = k * (k - 1) // 2 if bt in ("馬連", "ワイド") else k * (k - 1) * (k - 2) // 6
            elif kind == "軸1頭ながし":
                n = len(others) * (len(others) - 1) // 2
            elif kind in ("ながし", "軸2頭ながし"):
                n = len(others)
            else:
                n = 1
            out.append({"bet_type": bt, "kind": kind, "axis": axis, "others": others, "n": n, "stake": stake})
    return out
