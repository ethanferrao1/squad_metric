"""Live-AI plumbing for the page: placeholders that fill as results land.

The OpenRouter key is the viewer's own, typed in on the My Team page. It lives in
st.session_state only and is handed to live_ai.resolve() as an argument.
"""
import streamlit as st

import core
import live_ai
from ui import style as s

KEY = 'openrouter_key'
LABEL = {'live': ('live', s.ACCENT), 'cached': ('cached', s.MUTED),
         'template': ('template', s.FAINT), 'headlines only': ('headlines only', s.FAINT)}
REJECTED_NOTE = ('OpenRouter rejected that API key, so no live AI calls are being made. '
                 'Showing cached or template text instead. Check the key and paste it again.')


def api_key():
    """This browser session's key, or None."""
    return (st.session_state.get(KEY) or '').strip() or None


def budget():
    return st.session_state.setdefault('ai_budget', {'used': 0})


def rejected():
    return bool(budget().get('rejected'))


def enabled():
    return (st.session_state.get('live_ai', False) and not core.OFFLINE
            and bool(api_key()) and not rejected())


def spinner(text):
    return f"<span class='sm-spin'></span><span class='sm-muted'>{text}</span>"


def label(result):
    name, colour = LABEL.get(result.get('label'), LABEL['template'])
    return s.badge(name, colour)


class Slots:
    """Placeholders keyed by job; each redraws itself when one of its jobs lands."""

    def __init__(self):
        self.results, self.redraws = {}, {}

    def add(self, keys, draw):
        box = st.empty()
        redraw = lambda: box.markdown(draw(self.results), unsafe_allow_html=True)
        redraw()
        for k in keys:
            self.redraws.setdefault(k, []).append(redraw)

    def land(self, key, result):
        self.results[key] = result
        for redraw in self.redraws.get(key, []):
            redraw()

    def run(self, jobs, gw):
        """Resolve the jobs this page is showing, 5 at a time, filling slots as they land."""
        jobs = [j for j in jobs if j['key'] in self.redraws]
        was_rejected = rejected()
        live_ai.resolve(jobs, self.land, budget(), gw, live=enabled(),
                        memo=st.session_state.setdefault('ai_memo', {}),
                        api_key=api_key() if enabled() else None)
        if rejected() and not was_rejected:
            st.warning(REJECTED_NOTE)
