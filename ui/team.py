"""The tabs about the loaded team: My Team, Lineup, Transfers."""
import streamlit as st

import core
import images
from ui import ai, charts, pitch, player, style as s

VERDICT_ORDER = {'injury cover': 0, 'substitute instead': 0, 'strong': 1,
                 'hold': 2, 'below margin': 3}


def _p(code):
    return core.players().get(code, {})


def _name(code):
    return s.esc(_p(code).get('name', code))


def _next_gw(t):
    return {c: t['base'].get(c, 0.0) * t['play'].get(c, 0.0) for c in t['codes']}


def _no_history():
    s.card("<div class='sm-muted'>No gameweek histories are cached, so there is "
           "no next-gameweek projection. Run <code>python refresh.py</code>.</div>")


def my_team(t, pool, offline):
    r, cur = t['r'], t.get('current')
    c1, c2, c3 = st.columns(3)
    c1.metric('Team rating', s.pct(r['pct']),
              help='Share of what the best squad on the same money would score')
    if not cur:
        _no_history()
        return
    c2.metric('Captain pick', _p(cur['captain']).get('name', ''))
    c3.metric('Projected XI · next GW', s.pts(t['lineup']['current']))
    s.section('Your XI')
    picked = pitch.pitch(cur['xi'], cur['bench'], _next_gw(t),
                         captain=cur['captain'], vice=cur['vice'], key='team')
    st.caption('Projections are for next gameweek. Pick a name for details.')
    if picked:
        player.show(picked, pool, t['proj'], offline)


def lineup(t, pool, offline):
    lu = t['lineup']
    if lu is None:
        _no_history()
        return
    gw = t['gw'] + 1
    best, cur, steps = lu['best'], t['current'], lu['steps']
    c1, c2, c3 = st.columns(3)
    c1.metric(f'Projected gain · GW{gw}', s.pts(lu['total'], True))
    c2.metric('Your lineup', s.pts(lu['current']))
    c3.metric('Best lineup', s.pts(best['points']))
    st.caption("One gameweek only, on the availability-adjusted projection with "
               "FPL's auto-sub and vice-captain rules. Never added to transfer gains.")

    ruled_out = [c for c in cur['xi'] if t['play'].get(c, 0.0) == 0]
    if lu['total'] < 0.05 and not ruled_out:
        s.card(f"<div class='sm-card-title'>Your lineup is already optimal for "
               f"GW{gw}.</div><div class='sm-muted'>No substitution, armband or "
               f"bench change adds anything.</div>")
        return

    if lu['subs']:
        rows = ''.join(
            f"<div class='sm-spread' style='margin-top:10px'><div>Start "
            f"<b>{_name(i)}</b> for <b>{_name(o)}</b>"
            + (f"<div class='sm-muted'>{_name(o)} cannot play. An auto-sub would "
               f"cover him, which is why the gain is small, but only if the bench "
               f"order allows it.</div>" if o in ruled_out else '')
            + f"</div>{s.badge(s.pts(g, True), s.ACCENT)}</div>"
            for i, o, g in lu['subs'])
        s.section('Substitutions')
        s.card(f"<div class='sm-spread'><div class='sm-card-title'>Suggested subs</div>"
               f"{s.badge(s.pts(steps['subs'], True), s.ACCENT)}</div>{rows}"
               f"<div class='sm-muted' style='margin-top:12px'>Bench after these: "
               f"{' → '.join(_name(c) for c in best['bench'])}</div>")

    changes = []
    if best['captain'] != cur['captain']:
        changes.append(('Captain', cur['captain'], best['captain'], steps['captain']))
    if best['vice'] != cur['vice']:
        changes.append(('Vice-captain', cur['vice'], best['vice'], steps['vice']))
    if changes:
        s.section('Armband')
        s.card(''.join(f"<div class='sm-spread' style='margin:6px 0'><div>{k}: "
                       f"{_name(a)} → <b>{_name(b)}</b></div>"
                       f"{s.badge(s.pts(g, True), s.ACCENT)}</div>"
                       for k, a, b, g in changes))

    if steps['bench'] >= 0.05:
        s.section('Bench order')
        s.card(f"<div class='sm-spread'><div><b>"
               f"{' → '.join(_name(c) for c in best['bench'])}</b>"
               f"<div class='sm-muted'>Now: "
               f"{' → '.join(_name(c) for c in cur['bench'])}</div></div>"
               f"{s.badge(s.pts(steps['bench'], True), s.ACCENT)}</div>")

    involved = [c for i, o, _ in lu['subs'] for c in (i, o)]
    involved += [c for _, a, b, _ in changes for c in (a, b)]
    if steps['bench'] >= 0.05:
        involved += best['bench']
    if involved:
        st.caption('Pick a player for details.')
        picked = pitch.picker(list(dict.fromkeys(involved)), 'lineup-changes')
        if picked:
            player.show(picked, pool, t['proj'], offline)

    with st.expander('Suggested lineup on the pitch'):
        picked = pitch.pitch(best['xi'], best['bench'], _next_gw(t),
                             captain=best['captain'], vice=best['vice'],
                             key='lineup')
        if picked:
            player.show(picked, pool, t['proj'], offline)


def _sub_line(t, out_code):
    """The free fix for a player who cannot play: the Lineup tab's own answer."""
    cover = {o: i for i, o, _ in (t.get('lineup') or {}).get('subs', [])}
    if out_code in cover:
        return f'start {_name(cover[out_code])} in his place'
    subs = ', '.join(_name(c) for c in sorted(t['r'].get('sub_ins', ())))
    return f'start {subs} in his place' if subs else 'reshuffle the XI around him'


def _ai_text(res):
    """An explanation, or a spinner while it is on its way."""
    if res is None:
        return ai.spinner('Writing the explanation…')
    return f"{s.esc(res['text'])} {ai.label(res)}"


def _explain(row, t, out_code, res):
    from form_lab import form as fform
    gain, h = row['gain_pts'], fform.SWAP_HORIZON
    if row['verdict'] == 'substitute instead':
        return (f"Make the substitution instead: {_sub_line(t, out_code)} (free). "
                f"The best transfer adds only {s.pts(gain, True)} over that across "
                f"{h} gameweeks, under the {s.pts(fform.SWITCH_MARGIN)} margin.")
    text = _ai_text(res)
    if row['verdict'] == 'injury cover':
        text += (f"<div class='sm-muted' style='margin-top:6px'>That is "
                 f"{s.pts(gain, True)} more than the free option: "
                 f"{_sub_line(t, out_code)}.</div>")
    return text


def _side(code, name):
    p = _p(code)
    return (f"<div class='sm-row' style='gap:10px'><img class='avatar' "
            f"src='{images.photo(p, 112)}'><div><div class='sm-row' style='gap:6px'>"
            f"<img class='crest' src='{images.badge(p, 44)}'><b>{s.esc(name)}</b></div>"
            f"<div class='sm-muted'>{s.esc(p.get('club', ''))} · "
            f"{s.money(p.get('price', 0))}</div></div></div>")


PLAN_LABEL = ('Multi-transfer plans matched single swaps in a 3-season backtest; '
              'use them for flexibility.')


def _blend(t, code):
    b = t['proj'].drop_duplicates('code').set_index('code')['blended_ppg']
    return float(b[code]) if code in b.index else None


def _swap_card(row, i, t, pool, offline, code_of, slots):
    from form_lab import form as fform
    out_code, in_code = code_of.get(row['out']), code_of.get(row['in'])
    news = ' '.join(b for b in (s.sentiment_badge(core.verdict(int(c)))
                                for c in (out_code, in_code) if c is not None) if b)
    fpl = _p(out_code).get('news')
    key = f"{t['gw']}|swap|{row['out']}|{row['in']}"
    slots.add([key], lambda r: (
        f"<div class='sm-card'><div class='sm-spread'><div class='sm-row'>"
        f"{_side(out_code, row['out'])}<span class='arrow'>→</span>"
        f"{_side(in_code, row['in'])}</div><div class='sm-row'>"
        f"{s.verdict_badge(row['verdict'])}{s.badge(row['pos'])}</div></div>"
        + s.stats([('Cost', s.money_signed(row['cost'])),
                   (f'{fform.SWAP_HORIZON}-GW gain', s.pts(row['gain_pts'], True))])
        + f"<div class='sm-body'>{_explain(row, t, out_code, r.get(key))}</div>"
        + (f"<div class='sm-muted' style='margin-top:8px'>FPL: {s.esc(fpl)}</div>"
           if fpl else '')
        + (f"<div class='sm-row' style='margin-top:10px'>{news}</div>" if news else '')
        + '</div>'))
    c1, c2 = st.columns([3, 1])
    with c1:
        picked = pitch.picker([int(out_code), int(in_code)], f'swap{i}')
    if c2.toggle('Compare', key=f'cmp{i}'):
        s.chart(st, charts.points_chart(
            core.match_log(int(out_code), offline), _blend(t, out_code), row['out'],
            compare=(core.match_log(int(in_code), offline), _blend(t, in_code),
                     row['in'])))
    if picked:
        player.show(picked, pool, t['proj'], offline)


def _plan_card(n, p, best, t, pool, offline, slots):
    from form_lab import backtest as fbt
    hit = fbt.HIT_COST
    net = p['gain'] - hit * p['hits']
    moved = sum(_p(c).get('price', 0) for c in p['outs'])
    title = 'Keep the squad' if n == 0 else f"{n} transfer{'s' * (n > 1)}"
    tags = (s.badge('Best', s.ACCENT) if best else '') + \
        (s.badge(f"{p['hits']} hit{'s' * (p['hits'] > 1)} (−{hit * p['hits']} pts)",
                 s.WARN) if p['hits'] else '')
    pairs = core.plan_pairs(p['outs'], p['ins'])
    keys = [f"{t['gw']}|plan{n}|{_p(o).get('name')}|{_p(i).get('name')}" for o, i in pairs]
    slots.add(keys, lambda r: (
        f"<div class='sm-card{' best' if best else ''}'><div class='sm-spread'>"
        f"<div class='sm-card-title'>{title}</div><div class='sm-row'>{tags}</div></div>"
        + ''.join(f"<div class='sm-row' style='margin-top:12px'>{_side(o, _p(o).get('name'))}"
                  f"<span class='arrow'>→</span>{_side(i, _p(i).get('name'))}</div>"
                  f"<div class='sm-body' style='margin-top:6px'>{_ai_text(r.get(k))}</div>"
                  for (o, i), k in zip(pairs, keys))
        + s.stats([('Money moved', s.money(moved)), ('Net cost', s.money_signed(p['cost'])),
                   ('Net 5-GW gain', s.pts(net, True))]) + '</div>'))
    if n:
        picked = pitch.picker([*p['outs'], *p['ins']], f'plan{n}')
        if picked:
            player.show(picked, pool, t['proj'], offline)


def _free_hit(t, pool, offline):
    fh = core.free_hit_now(t['entry'], offline)
    if not fh:
        return
    n = len(fh['blank'])
    if not fh['play']:
        s.card(f"<div class='sm-muted'>Free hit for GW{fh['gw']}: not triggered. "
               f"{n} of your starters {'has' if n == 1 else 'have'} no fixture; it needs "
               f"{core.FH_MIN_BLANK} and a projected gain of {s.pts(core.FH_THRESHOLD)}.</div>")
        return
    s.section(f"Free hit for GW{fh['gw']}")
    s.card(f"<div class='sm-spread'><div><b>{n} of your starters have no fixture</b>"
           f"<div class='sm-muted'>{', '.join(_name(c) for c in fh['blank'])}</div></div>"
           f"{s.badge(s.pts(fh['gain'], True) + ' this GW', s.ACCENT)}</div>"
           f"<div class='sm-muted' style='margin-top:10px'>A one-week squad; your team "
           f"returns the week after. Rule adopted from the backtest (+110 over 3 seasons).</div>")
    bench = [c for c in fh['squad'] if c not in fh['xi']]
    bench.sort(key=lambda c: _p(c).get('element_type') != 1)
    picked = pitch.pitch(fh['xi'], bench, fh['values'], key='freehit')
    if picked:
        player.show(picked, pool, t['proj'], offline)


def plan_inputs(t):
    """The free-transfer and hits inputs, shared by Transfers and News."""
    guess = core.free_transfers(int(t['entry']), t['gw'])
    free = st.session_state.get('free_transfers', guess or 1)
    return int(free), int(st.session_state.get('allow_hits', 0)), guess


def transfers(t, pool, offline):
    from form_lab import backtest as fbt, form as fform
    if t['proj'] is None:
        _no_history()
        return
    _, _, guess = plan_inputs(t)
    c1, c2, c3 = st.columns(3)
    free = c1.number_input('Free transfers', 0, fbt.MAX_FREE, value=guess or 1,
                           key='free_transfers',
                           help='Estimated from your transfer history.' if guess
                           else 'Not available from the API; defaulting to 1.')
    hits = c2.number_input('Allow hits', 0, fbt.MAX_HITS, value=0, key='allow_hits')
    plans, best_n = core.transfer_plans(t['entry'], offline, int(free), int(hits))
    best = plans[best_n]
    c3.metric(f'Projected gain · next {fform.SWAP_HORIZON} GWs',
              s.pts(best['gain'] - fbt.HIT_COST * best['hits'], True),
              help='The best plan, net of hits. Never added to the lineup gain.')
    st.caption(f'Scored over {fform.SWAP_HORIZON} gameweeks on the availability-adjusted '
               f'blend; each transfer must clear {s.pts(fform.SWITCH_MARGIN)} and each '
               f'hit costs {fbt.HIT_COST} pts. Explanations and news are advisory and '
               f'never change a gain.')

    _free_hit(t, pool, offline)
    code_of = pool.drop_duplicates('web_name').set_index('web_name')['code']
    slots = ai.Slots()
    swaps = t['swaps']
    if len(swaps):
        swaps = swaps.assign(_o=swaps['verdict'].map(VERDICT_ORDER)) \
            .sort_values(['_o', 'gain_pts'], ascending=[True, False])
    cover = swaps[swaps['verdict'].isin(['injury cover', 'substitute instead'])] \
        if len(swaps) else swaps
    if len(cover):
        s.section('Injury cover')
        for i, (_, row) in enumerate(cover.iterrows()):
            _swap_card(row, f'c{i}', t, pool, offline, code_of, slots)

    s.section('Plans by number of transfers')
    st.caption(PLAN_LABEL)
    for n, p in plans.items():
        _plan_card(n, p, n == best_n, t, pool, offline, slots)

    rest = swaps.drop(cover.index) if len(swaps) else swaps
    s.section('Single swaps')
    if not len(rest):
        s.card("<div class='sm-muted'>No single transfer improves this squad.</div>")
    for i, (_, row) in enumerate(rest.iterrows()):
        _swap_card(row, f's{i}', t, pool, offline, code_of, slots)
    slots.run(core.explain_jobs(t, pool, plans), t['gw'])
