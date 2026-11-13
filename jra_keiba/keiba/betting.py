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


def order_probs(p: np.ndarray, lam2: float = 0.85, lam3: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Return (P2, P3): P2[i,j]=P(i 1st, j 2nd), P3[i,j,k]=P(i 1st, j 2nd, k 3rd)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(p)
    a = p ** lam2
    b = p ** lam3
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


def bet_probabilities(p: np.ndarray, numbers: list[int], lam2: float = 0.85, lam3: float = 0.75
                      ) -> dict[str, dict[tuple[int, ...], float]]:
    """Probability of every ticket of every bet type. Keys are tuples of 馬番.

    Unordered bet types use ascending tuples; ordered types (馬単, 三連単) use
    finishing order.
    """
    n = len(p)
    P2, P3 = order_probs(p, lam2, lam3)
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
    """Fit the qualifying tickets into a budget.

    Tickets are taken in order of hit probability, one unit each, until the money runs
    out; money left over after every ticket has one unit is added from the top, one
    unit at a time. Trimming by probability rather than by expected value keeps the
    backtested return about the same while hitting far more often (in the backtest the
    ten likeliest qualifying tickets hit 5.0% of the time against 1.7% for the ten
    with the highest expected value).
    """
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
