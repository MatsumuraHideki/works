import itertools

import numpy as np

from keiba import betting


def test_probabilities_are_consistent():
    p = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
    nums = list(range(1, 9))
    pr = betting.bet_probabilities(p, nums, 0.85, 0.75)
    for t, total in {"単勝": 1, "複勝": 3, "馬連": 1, "ワイド": 3, "馬単": 1, "三連複": 1, "三連単": 1}.items():
        assert abs(sum(pr[t].values()) - total) < 1e-9, t
    _, P3 = betting.order_probs(p, 0.85, 0.75)
    brute = sum(P3[i, j, k] for i, j, k in itertools.permutations(range(8), 3) if {0, 1} <= {i, j, k})
    assert abs(brute - pr["ワイド"][(1, 2)]) < 1e-12
    brute = sum(P3[i, j, k] for i, j, k in itertools.permutations(range(8), 3) if 0 in (i, j, k))
    assert abs(brute - pr["複勝"][(1,)]) < 1e-12


def test_place_slots_follow_jra_rules():
    assert [betting.place_slots(n) for n in (4, 5, 7, 8, 18)] == [0, 2, 2, 3, 3]
    pr = betting.bet_probabilities(np.full(6, 1 / 6), list(range(1, 7)))
    assert abs(sum(pr["複勝"].values()) - 2) < 1e-9


def test_selection_and_sizing():
    probs = {"単勝": {(1,): 0.5, (2,): 0.3, (3,): 0.001}}
    odds = {"単勝": {(1,): 1.8, (2,): 5.0, (3,): 2000.0}}
    bets = betting.evaluate_bets(probs, odds)
    picked = betting.select_bets(bets, min_ev=1.15)
    assert [b.combo for b in picked] == [(2,)]          # (1) is -EV, (3) is below the probability floor
    sized = betting.size_bets(picked, bankroll=100_000, kelly_fraction=0.25, max_race_fraction=0.05)
    assert sized[0].stake == 3100                       # kelly 0.125 * 0.25 * 100k = 3125 -> 100-yen units
    assert sized[0].stake % 100 == 0


def test_budget_slip_trims_by_probability_and_spends_the_budget():
    mk = lambda combo, prob, odds: betting.Bet("ワイド", combo, prob, odds, prob * odds, 0.0)
    bets = [mk((1, 2), 0.10, 12), mk((1, 3), 0.02, 80), mk((2, 3), 0.05, 25)]
    slip = betting.budget_slip(list(bets), 200)
    assert [(b.combo, b.stake) for b in slip] == [((1, 2), 100), ((2, 3), 100)]      # likeliest first
    slip = betting.budget_slip(list(bets), 500)
    assert [(b.combo, b.stake) for b in slip] == [((1, 2), 200), ((2, 3), 200), ((1, 3), 100)]
    assert sum(b.stake for b in slip) == 500
    assert betting.budget_slip([], 500) == []


def _expand(groups, three=False):
    from itertools import combinations
    out = []
    for kind, axis, others in groups:
        if kind == "ボックス":
            out += list(combinations(sorted(others), 3 if three else 2))
        elif kind == "ながし":
            out += [tuple(sorted((axis[0], o))) for o in others]
        elif kind == "軸2頭ながし":
            out += [tuple(sorted(axis + [o])) for o in others]
        elif kind == "軸1頭ながし":
            out += [tuple(sorted((axis[0],) + c)) for c in combinations(others, 2)]
        else:
            out.append(tuple(sorted(others)))
    return out


def test_grouping_covers_exactly_the_same_tickets():
    import random
    from itertools import combinations
    rng = random.Random(3)
    for _ in range(40):
        horses = list(range(1, 15))
        pairs = rng.sample(list(combinations(horses, 2)), rng.randint(1, 30))
        got = _expand(betting.group_pairs(pairs))
        assert sorted(got) == sorted(pairs) and len(got) == len(set(got))
        triples = rng.sample(list(combinations(horses, 3)), rng.randint(1, 25))
        got = _expand(betting.group_triples(triples), three=True)
        assert sorted(got) == sorted(triples) and len(got) == len(set(got))
    assert betting.group_pairs([(1, 2), (1, 3), (2, 3)]) == [("ボックス", [], [1, 2, 3])]
    assert betting.group_pairs([(5, 7), (5, 11), (5, 10)]) == [("ながし", [5], [7, 10, 11])]
    assert betting.group_triples([(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]) == [("ボックス", [], [1, 2, 3, 4])]
