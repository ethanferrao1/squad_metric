"""A squad on a pitch: shirts on striped grass by position, the bench below."""
import streamlit as st

import core
import images
from ui import style as s

# text badges: the circled-letter emoji render as blank circles in some fonts
MARKER = {'C': 'C', 'VC': 'V'}


def _player(code, value, unit, marker):
    p = core.players().get(code, {})
    mk = (f"<div class='mk{' vc' if marker == 'VC' else ''}'>{MARKER[marker]}</div>"
          if marker else '')
    return (f"<div class='pl'>{mk}<img src='{images.shirt(p)}' alt=''>"
            f"<div class='nm'>{s.status_dot(p.get('status', 'a'))}"
            f"{s.esc(p.get('name', code))}</div>"
            f"<div class='pj'>{s.num(value)} {unit}</div></div>")


def html(xi, bench, values, unit='pts', captain=None, vice=None):
    """The whole pitch and bench as one block of HTML."""
    pos = {c: core.players().get(c, {}).get('element_type', 0) for c in [*xi, *bench]}
    marks = {captain: 'C', vice: 'VC'}
    rows = []
    for p in (1, 2, 3, 4):
        row = sorted((c for c in xi if pos[c] == p), key=lambda c: -(values.get(c) or 0))
        rows.append("<div class='prow'>" + ''.join(
            _player(c, values.get(c), unit, marks.get(c)) for c in row) + '</div>')
    bench_row = ''.join(_player(c, values.get(c), unit, marks.get(c)) for c in bench)
    return (f"<div class='pitch'>{''.join(rows)}</div>"
            f"<div class='bench'><div class='bench-label'>BENCH</div>"
            f"<div class='prow'>{bench_row}</div></div>")


def _picked(key):
    """Remember the name just clicked, and clear the chips for the next click."""
    chosen = st.session_state.get(key) or []
    st.session_state[f'{key}-open'] = chosen[-1] if chosen else None
    st.session_state[key] = []


def picker(codes, key):
    """Name chips for these players; returns the one clicked, once per click."""
    names = {core.players().get(c, {}).get('name', str(c)): c for c in codes}
    chips = f'{key}-pick'
    st.pills('Player details', list(names), selection_mode='multi', key=chips,
             on_change=_picked, args=(chips,), label_visibility='collapsed')
    return names.get(st.session_state.pop(f'{chips}-open', None))


def pitch(xi, bench, values, unit='pts', captain=None, vice=None, key='pitch'):
    """Draw the pitch; returns a player picked for details, once per click."""
    st.markdown(html(xi, bench, values, unit, captain, vice), unsafe_allow_html=True)
    return picker([*xi, *bench], key)
