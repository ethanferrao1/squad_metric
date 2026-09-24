"""Squadmetric: evidence-led squad and transfer analytics for FPL.

    python refresh.py        # or the Refresh data button, in the background
    streamlit run app.py

Layout only: every number comes from core.py, the lineup optimiser from
lineup.py, live AI from live_ai.py. Only the chosen section runs, so the AI is
called for Transfers or News only when that section is open. FPL_OFFLINE=1
for a demo that never touches the network. Live AI uses the viewer's own
OpenRouter key, held in session state only. The team ID and key are set on
the My Team page, which folds them away once both are settled.
"""
import logging
import traceback

import streamlit as st

import core
import llm_explain
import refresher
from ui import ai, build, player, pool as pool_tabs, style as s, team as team_tabs

TABS = ['My Team', 'Lineup', 'Transfers', 'Players', 'News',
        'Build']
TEAM_TABS = TABS[:3]


def load_message(entry, error):
    """What to tell the user when a team cannot be loaded."""
    code = getattr(error, 'code', None)
    if code == 404:
        return f'No FPL team has ID {entry}. Check the number in your FPL points URL.'
    if isinstance(error, core.NoPicks):
        return str(error)
    if isinstance(error, (OSError, FileNotFoundError)):
        return (f'Team {entry} could not be fetched and is not cached. '
                f'Check the connection, or try a cached team.')
    return f'Team {entry} could not be analysed ({type(error).__name__}).'


def log_failure(message):
    """logging.exception, with this session's key and anything key-shaped masked."""
    logging.error(llm_explain.redact(f'{message}\n{traceback.format_exc()}', ai.api_key()))


# Setup lives on the My Team page and folds away once a team has loaded and
# the AI is settled: a key OpenRouter accepted, "without AI", or offline.

def _start_refresh():
    started, message = refresher.start()
    st.session_state['refresh_toast'] = (
        message, ':material/sync:' if started else ':material/info:')


def refresh_button():
    st.button('Refresh data', key='refresh', icon=':material/sync:',
              disabled=core.OFFLINE, on_click=_start_refresh,
              help='Fetch the latest FPL data in the background. '
                   'The page keeps working meanwhile.')


def load():
    """(analysis or None, error text or None, [(st method, note)] for every page)."""
    loaded = st.session_state.get('entry', '')
    if not loaded:
        return None, None, []
    if not loaded.isdigit():
        return None, 'A team ID is a number.', []
    try:
        t = core.team_analysis(loaded, core.OFFLINE)
    except Exception as e:
        log_failure(f'team {loaded} failed to load')
        return None, load_message(loaded, e), []
    notes = [('info', t['chip_note'])] if t.get('chip_note') else []
    if t['fell_back']:
        notes.append(('warning', f"Offline: showing cached data (team as of GW{t['team_gw']})."))
    return t, None, notes


def ai_settled():
    return (core.OFFLINE or st.session_state.get('ai_skipped')
            or (bool(ai.api_key()) and not ai.rejected()))


def _submit(skip):
    """Store the form's team ID and, if given, a key OpenRouter accepts."""
    st.session_state['entry'] = st.session_state['setup_team'].strip()
    st.session_state.pop('setup_error', None)
    key = '' if skip else (st.session_state.get('setup_key') or '').strip()
    st.session_state['ai_skipped'] = skip or st.session_state.get('ai_skipped', False)
    if key:
        ok = llm_explain.check_key(key)
        if ok is False:
            st.session_state['setup_error'] = ('OpenRouter rejected that API key. '
                                               'Check it and try again.')
        else:
            st.session_state[ai.KEY] = key
            st.session_state['setup_key'] = ''
            st.session_state['ai_skipped'] = False
            if ok is None:
                st.session_state['setup_note'] = ("The key couldn't be checked just now; "
                                                  'it will be tried when AI is needed.')
    st.session_state['setup_open'] = False


def setup_card(error):
    """Team ID and optional OpenRouter key, in one form."""
    with st.container(border=True, key='setup'):
        s.section('Load your team')
        with st.form('setup_form', border=False, enter_to_submit=True):
            st.text_input('FPL team ID', key='setup_team', placeholder='e.g. 3265946',
                          value=st.session_state.get('entry', ''),
                          help='The number in the URL of your FPL points page.')
            if not core.OFFLINE:
                has = bool(ai.api_key()) and not ai.rejected()
                st.text_input('OpenRouter API key (optional)', type='password',
                              key='setup_key', autocomplete='off',
                              placeholder='Key saved for this session' if has
                              else 'Paste your key')
                st.caption('Enables live AI explanations. Your key stays in this '
                           'browser session only. [Get a key](https://openrouter.ai/keys)')
            with st.container(horizontal=True):
                st.form_submit_button('Load team', type='primary', key='setup_load',
                                      on_click=_submit, args=(False,))
                if not core.OFFLINE:
                    st.form_submit_button('Continue without AI', key='setup_skip',
                                          on_click=_submit, args=(True,))
        for text in (error, st.session_state.get('setup_error')):
            if text:
                st.error(text)
        if ai.rejected():
            st.warning(ai.REJECTED_NOTE)


def _reopen_setup():
    st.session_state['setup_open'] = True


def _live_ai_changed():
    st.session_state['live_ai'] = st.session_state['live_ai_toggle']


def team_summary(t):
    """The folded setup: the team's money, the AI switch, and a way back in."""
    st.markdown(s.tiles([('Team', st.session_state['entry']),
                         ('Bank', s.money(t['r']['bank'])),
                         ('Team value', s.money(t['r']['spend']))]),
                unsafe_allow_html=True)
    with st.container(horizontal=True, vertical_alignment='center', key='teamrow'):
        if ai.api_key() and not core.OFFLINE:
            st.toggle('Live AI', key='live_ai_toggle', value=st.session_state['live_ai'],
                      on_change=_live_ai_changed,
                      help='Explanations and news ratings for your team. '
                           'Advisory only; never changes a suggestion.')
        else:
            st.caption('AI text is cached or template, as labelled.')
        st.button('Change team or key', key='setup_change', on_click=_reopen_setup)
    note = st.session_state.pop('setup_note', None)
    if note:
        st.caption(note)


def team_setup(t, error):
    """My Team's top: the setup form until it is settled, then the summary."""
    if t and ai_settled() and not st.session_state.get('setup_open'):
        team_summary(t)
    else:
        if t:
            st.markdown(s.tiles([('Bank', s.money(t['r']['bank'])),
                                 ('Team value', s.money(t['r']['spend']))]),
                        unsafe_allow_html=True)
        setup_card(error)


def _refresh_status():
    """The refresh progress line; on completion, a toast and a full rerun onto the new data."""
    state = refresher.STATE
    st.session_state.setdefault('refresh_seen', state['gen'])
    if state['running']:
        st.markdown(ai.spinner(refresher.status()), unsafe_allow_html=True)
    elif st.session_state['refresh_seen'] != state['gen']:
        st.session_state['refresh_seen'] = state['gen']
        if state['ok']:
            st.session_state['refresh_toast'] = (f"Data updated to GW{state['gw']}",
                                                 ':material/check_circle:')
            st.session_state['reopen_panel'] = True
            st.rerun()
        st.toast('Data refresh failed. Still showing the previous data.',
                 icon=':material/error:')


# ticks every 2 s only while a refresh runs, re-rendering just itself
_status_live = st.fragment(run_every=2)(_refresh_status)
_status_idle = st.fragment(_refresh_status)


def section():
    """The chosen section; only it is rendered."""
    wanted = st.query_params.get('tab')
    index = next((i for i, n in enumerate(TABS) if wanted and wanted.lower() in n.lower()), 0)
    return st.radio('Section', TABS, index=index, horizontal=True, key='section',
                    label_visibility='collapsed')


def render(name, t, pool, error):
    offline = core.OFFLINE
    if name == TABS[0]:
        team_setup(t, error)
        if t:
            team_tabs.my_team(t, pool, offline)
    elif name in TEAM_TABS and t is None:
        s.card("<div class='sm-card-title'>Load a team to begin</div>"
               "<div class='sm-muted' style='margin-top:6px'>Enter your FPL team ID "
               "on the My Team page.</div>")
    elif name == TABS[1]:
        team_tabs.lineup(t, pool, offline)
    elif name == TABS[2]:
        team_tabs.transfers(t, pool, offline)
    elif name == TABS[3]:
        proj = t['proj'] if t else core.blended(pool, (), offline)[0]
        pool_tabs.players(pool, proj, offline)
    elif name == TABS[4]:
        pool_tabs.news(pool, t, offline)
    else:
        build.render(t, pool, offline)


def main():
    st.set_page_config(page_title=s.BRAND, layout='wide')
    s.inject()
    try:
        pool = core.load_pool()
    except FileNotFoundError:
        st.error('No player pool yet. Run `python refresh.py` first.')
        return

    # after a refresh's rerun, the panel that was open (player.show/_closed track it)
    reopen = st.session_state.pop('reopen_panel', False)
    toast = st.session_state.pop('refresh_toast', None)
    if toast:
        st.toast(toast[0], icon=toast[1])

    if 'entry' not in st.session_state and st.query_params.get('team'):
        st.session_state['entry'] = st.query_params['team']
    st.session_state.setdefault('live_ai', True)
    # a changed key gets a fresh chance after a rejection; compared by hash
    seen = hash(ai.api_key() or '')
    if st.session_state.get('ai_key_hash') != seen:
        st.session_state['ai_key_hash'] = seen
        ai.budget().pop('rejected', None)

    gw, refreshed = core.data_stamp()
    with st.container(horizontal=True, vertical_alignment='top', key='topbar'):
        s.header(gw, refreshed, core.OFFLINE)
        refresh_button()
    (_status_live if refresher.STATE['running'] else _status_idle)()
    t, error, notes = load()
    for kind, text in notes:
        getattr(st, kind)(text)
    name = section()
    try:
        render(name, t, pool, error)
        if reopen:
            proj = t['proj'] if t else core.blended(pool, (), core.OFFLINE)[0]
            player.reopen(pool, proj, core.OFFLINE)
    except Exception:
        log_failure(f'section {name} failed')
        s.card("<div class='sm-card-title'>This section could not be shown</div>"
               "<div class='sm-muted' style='margin-top:6px'>Something in the data for "
               "it is missing or unexpected. The other sections still work; "
               "try Load team again or Refresh data.</div>")


if __name__ == '__main__':
    main()
