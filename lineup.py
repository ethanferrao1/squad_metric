"""Next-gameweek lineup: the best XI, bench order, captain and vice.

Expected points follow FPL's rules on the availability-adjusted projection:
a starter who does not play is replaced by the next bench player (the backup
keeper only for the keeper), and the vice takes the armband if the captain
does not play. Auto-subs ignore formation limits -- a small approximation.
Every legal XI is scored, so the result is exact under that model.
"""
import itertools

import optimise as op

OUTFIELD = (2, 3, 4)
TIE = 1e-9


def _at_least(probs):
    """tail[k] = P(at least k of these independent events happen)."""
    dist = [1.0]
    for q in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, p in enumerate(dist):
            nxt[k] += p * (1 - q)
            nxt[k + 1] += p * q
        dist = nxt
    return [sum(dist[k:]) for k in range(len(dist))]


def expected_points(xi, bench, captain, vice, base, play, pos):
    """Expected next-gameweek points of a lineup.

    `base` is what a player scores if he plays, `play` the chance he plays.
    """
    pts = {c: base.get(c, 0.0) * play.get(c, 0.0) for c in (*xi, *bench)}
    total = sum(pts[c] for c in xi)
    total += pts[captain] + (1 - play.get(captain, 0.0)) * pts.get(vice, 0.0)

    keeper = [c for c in xi if pos[c] == 1]
    spare = [c for c in bench if pos[c] == 1]
    if keeper and spare:
        total += (1 - play.get(keeper[0], 0.0)) * pts[spare[0]]

    tail = _at_least([1 - play.get(c, 0.0) for c in xi if pos[c] != 1])
    outfield = [c for c in bench if pos[c] != 1]
    total += sum(tail[i] * pts[c] for i, c in enumerate(outfield, 1)
                 if i < len(tail))
    return total


def armband(xi, base, play):
    """(captain, vice) maximising the captain bonus; never an unavailable
    captain. With a certain starter as captain the vice is worth nothing, so
    ties go to the stronger vice."""
    pts = {c: base.get(c, 0.0) * play.get(c, 0.0) for c in xi}
    captains = [c for c in xi if play.get(c, 0.0) > 0] or list(xi)
    return max(((c, v) for c in captains for v in xi if v != c),
               key=lambda cv: (pts[cv[0]]
                               + (1 - play.get(cv[0], 0.0)) * pts[cv[1]],
                               pts[cv[1]]))


def legal(xi, pos):
    """1 GK and 3-5 DEF, 2-5 MID, 1-3 FWD, eleven in all."""
    counts = {p: sum(pos[c] == p for c in xi) for p in (1, *OUTFIELD)}
    return (len(xi) == 11 and counts[1] == 1 and
            all(op.XI_MIN[p] <= counts[p] <= op.XI_MAX[p] for p in OUTFIELD))


def best(squad, base, play, pos):
    """The lineup with the highest expected points, as a dict.

    Nobody ruled out of next gameweek starts, unless no legal XI exists
    without him. (Under the auto-sub rule starting him can score the same,
    but a lineup should not rely on it.)
    """
    keepers = [c for c in squad if pos[c] == 1]
    outfield = [c for c in squad if pos[c] != 1]
    for allow_out in (False, True):
        top, top_score = None, -1e18
        for gk in keepers:
            for combo in itertools.combinations(outfield, 10):
                xi = [gk, *combo]
                if not legal(xi, pos) or (not allow_out and any(
                        play.get(c, 0.0) == 0 for c in xi)):
                    continue
                cap, vice = armband(xi, base, play)
                rest = [c for c in outfield if c not in combo]
                for order in itertools.permutations(rest):
                    bench = [k for k in keepers if k != gk] + list(order)
                    score = expected_points(xi, bench, cap, vice, base, play,
                                            pos)
                    if score > top_score:
                        top_score = score
                        top = {'xi': xi, 'bench': bench, 'captain': cap,
                               'vice': vice}
        if top:
            top['points'] = top_score
            return top
    raise ValueError('no legal XI in this squad')


def _keep_ties(opt, current, e, base, play, pos):
    """Where the manager's own captain, vice or bench order scores as well
    as the best, keep it -- a change worth 0.0 is not advice. Players new to
    the bench go after his, strongest first."""
    for key in ('captain', 'vice'):
        alt = {**opt, key: current[key]}
        if (alt[key] in opt['xi'] and alt['captain'] != alt['vice']
                and play.get(alt['captain'], 0.0) > 0
                and e(alt) >= e(opt) - TIE):
            opt = alt
    order = [c for c in current['bench'] if c in opt['bench']]
    order += sorted((c for c in opt['bench'] if c not in order),
                    key=lambda c: -base.get(c, 0.0) * play.get(c, 0.0))
    alt = {**opt, 'bench': sorted(order, key=lambda c: pos[c] != 1)}
    return alt if e(alt) >= e(opt) - TIE else opt


def _pairs(outs, ins, pos):
    """Match each player leaving the XI with one coming in, same position first."""
    ins, pairs = list(ins), []
    for o in outs:
        i = next((c for c in ins if pos[c] == pos[o]), ins[0])
        ins.remove(i)
        pairs.append((i, o))
    return pairs


def compare(current, base, play, pos):
    """Steps from the current lineup to the best, each with its expected gain.

    Subs first (keeping the current armband and bench order where they still
    fit), then captain and vice, then bench order. The steps sum to `total`.
    """
    squad = [*current['xi'], *current['bench']]
    e = lambda l: expected_points(l['xi'], l['bench'], l['captain'],
                                  l['vice'], base, play, pos)
    opt = _keep_ties(best(squad, base, play, pos), current, e, base, play, pos)
    opt['points'] = e(opt)

    keep_cap = (current['captain'] in opt['xi']
                and play.get(current['captain'], 0.0) > 0)
    cap = current['captain'] if keep_cap else opt['captain']
    vice = current['vice'] if (current['vice'] in opt['xi']
                              and current['vice'] != cap) else \
        next(v for v in sorted(opt['xi'], key=lambda c: -base.get(c, 0.0)
                               * play.get(c, 0.0)) if v != cap)
    order = [c for c in [*current['bench'], *current['xi']]
             if c not in opt['xi']]
    bench = sorted(order, key=lambda c: pos[c] != 1)

    subs = {'xi': opt['xi'], 'bench': bench, 'captain': cap, 'vice': vice}
    capped = {**subs, 'captain': opt['captain'],
              'vice': subs['vice'] if subs['vice'] != opt['captain']
              else opt['vice']}
    armed = {**subs, 'captain': opt['captain'], 'vice': opt['vice']}
    now = e(current)
    steps = {'subs': e(subs) - now,
             'captain': e(capped) - e(subs),
             'vice': e(armed) - e(capped),
             'bench': opt['points'] - e(armed)}
    pts = lambda c: base.get(c, 0.0) * play.get(c, 0.0)
    outs = sorted(set(current['xi']) - set(opt['xi']), key=pts)
    ins = sorted(set(opt['xi']) - set(current['xi']), key=pts, reverse=True)

    def alone(i, o):
        """Expected gain of this one sub on its own, auto-subs included."""
        swap = {o: i, i: o}
        alt = {**current, 'xi': [swap.get(c, c) for c in current['xi']],
               'bench': [swap.get(c, c) for c in current['bench']]}
        if alt['captain'] == o or alt['vice'] == o:
            alt['captain'], alt['vice'] = subs['captain'], subs['vice']
        return e(alt) - now

    return {'current': now, 'best': opt, 'steps': steps,
            'total': opt['points'] - now,
            'subs': [(i, o, alone(i, o)) for i, o in _pairs(outs, ins, pos)]}
