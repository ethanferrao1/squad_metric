"""The player detail panel: photo and badge, key figures, form, price and drivers."""
import plotly.graph_objects as go
import streamlit as st

import build_live
import core
import images
from ui import charts, style as s


def _bars(x, y, title, colours=None, horizontal=False):
    kw = dict(y=x, x=y, orientation='h') if horizontal else dict(x=x, y=y)
    fig = go.Figure(go.Bar(**kw, marker_color=colours or s.ACCENT,
                           hovertemplate='%{x}: %{y:.1f}<extra></extra>'
                           if not horizontal else '%{y}: %{x:+.1f}<extra></extra>'))
    return s.plot_layout(fig, title)


def detail(code, pool, proj, offline):
    p = core.players().get(code)
    if p is None:
        st.warning('This player is not in the current FPL player list.')
        return
    st.markdown(
        f"<div class='sm-row'><img class='avatar-lg' src='{images.photo(p)}'>"
        f"<div><div class='sm-row'><img class='crest-lg' src='{images.badge(p, 44)}'>"
        f"<div><div class='sm-brand' style='font-size:26px'>{s.esc(p['name'])}</div>"
        f"<div class='sm-muted'>{s.esc(p['club'])} Â· "
        f"{s.POS.get(p['element_type'], '')} Â· {s.money(p['price'])}</div></div></div>"
        f"<div style='margin-top:10px'>{s.status_badge(p['status'], p['chance'])}</div>"
        + (f"<div class='sm-muted' style='margin-top:8px'>{s.esc(p['news'])}</div>"
           if p['news'] else '') + '</div></div>',
        unsafe_allow_html=True)

    hist = core.history(code, offline)
    last10 = hist.tail(10)
    pr = core.projection(pool, proj, code, offline)
    mult = core.avail_map([code]).get(code, [1.0] * 5)
    blend = pr['blended_ppg']
    st.markdown(s.stats([
        ('Season points', p['total_points']), ('Points per game', s.num(p['ppg'])),
        ('Form, last 10', s.pts(last10['points'].mean() if len(last10) else None)),
        ('Minutes', p['minutes']), ('Goals', p['goals']), ('Assists', p['assists']),
        ('Clean sheets', p['clean_sheets']), ('Bonus', p['bonus']),
        ('Price', s.money(p['price'])), ('Owned', s.pct(p['selected'])),
        ('Next GW', s.pts(None if blend is None else blend * mult[0])),
        ('Next 5 GWs', s.pts(None if blend is None else blend * sum(mult))),
    ]), unsafe_allow_html=True)

    s.chart(st, charts.points_chart(core.match_log(code, offline), blend, p['name']))
    c1, c2 = st.columns(2)
    if hist.empty:
        c1.caption('No price history cached' + (' (offline).' if offline else '.'))
    else:
        s.chart(c1, _price(hist))
    _drivers(c2, pool, code)

    c1, c2 = st.columns(2)
    shown = [(l, v) for l, v in zip(('Prior', 'Form', 'Blended'),
                                    (pr.get('prior_ppg'), pr.get('form_ppg'), blend))
             if v is not None and v == v]
    if shown:
        s.chart(c1, _bars([l for l, _ in shown], [v for _, v in shown],
                          'Points per gameweek: prior, form, blend',
                          [s.FAINT, s.MUTED, s.ACCENT][-len(shown):]))
    with c2:
        s.section('Next five gameweeks')
        fx = core.fixtures_ahead(p['team'], core.api.current_gw())
        st.markdown(''.join(s.fdr_chip(*f) for f in fx) or 'No fixtures listed.',
                    unsafe_allow_html=True)


def _price(hist):
    fig = go.Figure(go.Scatter(x=hist['gw'], y=hist['price'] / 10, mode='lines',
                               line=dict(color=s.ACCENT, width=2),
                               hovertemplate='GW%{x}: Â£%{y:.1f}m<extra></extra>'))
    fig = s.plot_layout(fig, 'Price')
    fig.update_yaxes(tickprefix='Â£', ticksuffix='m', tickformat='.1f', dtick=0.1)
    fig.update_xaxes(tickprefix='GW', dtick=1)
    return fig


def _drivers(where, pool, code):
    drivers = sorted(build_live.player_drivers(pool, code), key=lambda r: abs(r['shap']))
    if not drivers:
        where.caption('No season-model drivers: no Premier League history.')
        return
    s.chart(where, _bars([f"{r['feature']} ({s.num(r['value'])})" for r in drivers],
                         [r['shap'] for r in drivers], 'Season-model drivers',
                         [s.ACCENT if r['shap'] >= 0 else s.FAINT for r in drivers],
                         horizontal=True))


show = st.dialog('Player details', width='large')(detail)
