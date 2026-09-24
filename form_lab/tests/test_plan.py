"""The multi-transfer MILP, on a hand-made squad and pool."""
import pytest

import optimise as op
from form_lab import plan

MARGIN = 8.0

# code: (position, price, 5-GW value). The XI is 1-4-4-2 around the 30-point
# midfielder; the 5-point defender (12) and the 1-point men sit on the bench.
SQUAD = {1: (1, 45, 10), 2: (1, 40, 1),
         10: (2, 50, 12), 11: (2, 50, 10), 13: (2, 50, 10), 14: (2, 50, 10),
         12: (2, 40, 5),
         20: (3, 120, 30), 21: (3, 60, 10), 22: (3, 60, 10), 23: (3, 60, 10),
         24: (3, 45, 1),
         30: (4, 70, 10), 31: (4, 70, 10), 32: (4, 45, 1)}


def market(extra):
    """pos, clubs, prices, values for the squad plus `extra` candidates."""
    everyone = {**SQUAD, **extra}
    return dict(pos={c: p for c, (p, _, _) in everyone.items()},
                clubs={c: c for c in everyone},          # one club each
                prices={c: pr for c, (_, pr, _) in everyone.items()},
                value={c: v for c, (_, _, v) in everyone.items()})


def run(extra, bank=0, free=1, max_hits=2, **kw):
    m = market(extra)
    sells = {c: m['prices'][c] for c in SQUAD}
    return plan.transfer_plan(list(SQUAD), bank, sells, free, max_hits,
                              margin=MARGIN, **m, **kw)


def squad_after(p):
    return (set(SQUAD) - set(p['outs'])) | set(p['ins'])


def test_zero_transfers_is_always_an_option():
    best, by = run({})
    assert by[0]['transfers'] == 0 and by[0]['gain'] == 0
    assert best['transfers'] == 0


def test_every_plan_is_legal_and_within_budget():
    extra = {40: (3, 90, 25), 41: (2, 65, 18), 42: (4, 80, 20), 43: (1, 50, 14)}
    best, by = run(extra, bank=30, free=2)
    m = market(extra)
    for p in [best, *filter(None, by.values())]:
        after = squad_after(p)
        counts = {k: sum(m['pos'][c] == k for c in after) for k in op.SQUAD}
        assert len(after) == 15 and counts == op.SQUAD
        assert p['cost'] <= 30
        assert len(p['xi']) == 11 and p['xi'] <= after


def test_a_worthwhile_free_transfer_is_made():
    best, _ = run({40: (3, 60, 30)})              # +20 for a 10-point midfielder
    assert best['ins'] == [40] and best['hits'] == 0


def test_a_hit_is_not_taken_when_it_does_not_pay():
    """The second upgrade is worth +10: past the margin, short of margin + 4."""
    best, _ = run({40: (3, 60, 30), 41: (4, 70, 20)}, free=1)
    assert best['transfers'] == 1 and best['hits'] == 0


def test_a_hit_is_taken_when_it_pays():
    """Worth +15 now: clears margin + 4, so it is worth a hit."""
    best, _ = run({40: (3, 60, 30), 41: (4, 70, 25)}, free=1)
    assert best['transfers'] == 2 and best['hits'] == 1


def test_a_two_for_two_that_moves_money_is_found():
    """Expensive MID + cheap DEF out, mid-price DEF + mid-price MID in.

    Neither swap works alone: the MID for MID is worth nothing, and the DEF
    for DEF is unaffordable until the MID sale frees the money. Forward 31 is
    dropped to 1 point so the XI has no spare: selling the cheap bench DEF
    (+27) then strictly beats selling a starting DEF (+22).
    """
    extra = {40: (2, 75, 32), 41: (3, 85, 30), 31: (4, 70, 1)}
    best, _ = run(extra, bank=0, free=2)
    assert set(best['outs']) == {20, 12} and set(best['ins']) == {40, 41}
    assert best['cost'] <= 0
    assert best['gain'] == pytest.approx(27)

    alone, _ = run({41: extra[41], 31: extra[31]}, bank=0, free=2)
    assert alone['transfers'] == 0


def test_bench_fodder_must_be_a_regular_starter():
    """A cheap non-regular is not bought just to sit on the bench."""
    extra = {40: (3, 60, 30), 41: (3, 40, 0)}
    best, _ = run(extra, bank=0, free=2, eligible=set(SQUAD) | {40})
    assert 41 not in best['ins']


def test_the_wildcard_ignores_margin_and_hits():
    extra = {40: (3, 60, 30), 41: (4, 70, 20), 42: (2, 50, 16)}
    m = market(extra)
    sells = {c: m['prices'][c] for c in SQUAD}
    wc = plan.wildcard_plan(list(SQUAD), 0, sells, **m)
    normal, _ = run(extra, free=1)
    assert wc['transfers'] >= normal['transfers']
    assert plan.wildcard_gain(normal, wc) > 0
