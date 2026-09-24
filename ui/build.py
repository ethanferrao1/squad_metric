"""The Build tab: wildcard now, the season followed since GW1, and the pre-season squad."""
import plotly.graph_objects as go
import streamlit as st

import core
from form_lab import backtest as fbt
from ui import pitch, player, style as s

VIEWS = ['Wildcard now', 'Followed since GW1', 'Pre-season (reference)']


def _bench(squad, xi):
    rest = [c for c in squad if c not in set(xi)]
    return sorted(rest, key=lambda c: core.players().get(c, {}).get('element_type') != 1)


def _show(picked, pool, proj, offline):
    if picked:
        player.show(picked, pool, proj, offline)


def wildcard(t, pool, offline):
    if t is None or t['proj'] is None:
        s.card("<div class='sm-muted'>Load a team to see the squad a wildcard "
               "would build from it.</div>")
        return
    wc = core.wildcard_now(t['entry'], offline)
    c1, c2 = st.columns(2)
    c1.metric('5-GW value over your squad', s.pts(wc['gain'], True))
    c2.metric('Players changed', len(wc['ins']))
    st.caption('Informational: the best squad your money buys on the next five '
               'gameweeks. The backtest found the fixed GW16/35 wildcard as good '
               'as any trigger, so this is not advice to play it now.')
    m, _ = core._market(pool, t)
    _show(pitch.pitch(wc['xi'], _bench(wc['squad'], wc['xi']), m['value'],
                      unit='5-GW pts', key='wc'), pool, t['proj'], offline)


def followed(pool, offline):
    v = core.build_views(offline)
    if not v:
        s.card("<div class='sm-muted'>No finished gameweek yet.</div>")
        return
    g = v['per_gw']
    c1, c2, c3 = st.columns(3)
    c1.metric('Followed, total', s.pts(g['followed'].sum()))
    c2.metric('Pre-season squad held', s.pts(g['held'].sum()))
    c3.metric('FPL average manager', s.pts(g['average'].sum()))
    st.caption(f"The live season to GW{v['gw']}, replayed with the backtest's logic from "
               f"the pre-season squad: greedy swaps, the {s.pts(fbt.form.SWITCH_MARGIN)} "
               f"margin, hits. "
               f"Chips are left out until their windows close. Every decision is "
               f"checked to use only data from before its deadline.")
    fig = go.Figure()
    for col, name, colour, dash in (('followed', 'Followed', s.ACCENT, None),
                                    ('held', 'Held', s.MUTED, None),
                                    ('average', 'FPL average', s.FAINT, 'dash')):
        fig.add_trace(go.Scatter(x=g['gw'], y=g[col].cumsum(), name=name, mode='lines+markers',
                                 line=dict(color=colour, width=2, dash=dash),
                                 hovertemplate='GW%{x}: %{y:.0f} pts<extra>' + name + '</extra>'))
    fig = s.plot_layout(fig, 'Cumulative points', height=300)
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.2, x=0))
    fig.update_xaxes(tickprefix='GW', dtick=1)
    s.chart(st, fig)

    s.section('Transfer log')
    if not v['log']:
        s.card("<div class='sm-muted'>No transfer has cleared the margin yet.</div>")
    name = lambda c: s.esc(core.players().get(c, {}).get('name', c))
    for gw, outs, ins in v['log']:
        s.card(f"<div class='sm-spread'><b>GW{gw}</b><div>"
               f"{', '.join(name(c) for c in outs)} <span class='arrow'>→</span> "
               f"{', '.join(name(c) for c in ins)}</div></div>")


def preseason(pool, offline):
    v = core.build_views(offline)
    if not v:
        s.card("<div class='sm-muted'>No season data yet.</div>")
        return
    squad = list(v['preseason'])
    pred = pool.drop_duplicates('code').set_index('code')['pred']
    values = {c: float(pred.get(c, 0)) for c in squad}
    pos = {c: core.players().get(c, {}).get('element_type') for c in squad}
    xi = sorted(fbt.xi_members(squad, values, pos))
    st.caption('Season model, bought at GW1 prices for £100.0m: the squad the '
               '"followed" replay starts from.')
    _show(pitch.pitch(xi, _bench(squad, xi), values, unit='season pts', key='pre'),
          pool, None, offline)


def render(t, pool, offline):
    s.section('Build')
    view = st.radio('View', VIEWS, horizontal=True, key='build_view',
                    label_visibility='collapsed')
    if view == VIEWS[1]:
        followed(pool, offline)
    elif view == VIEWS[2]:
        preseason(pool, offline)
    else:
        wildcard(t, pool, offline)
