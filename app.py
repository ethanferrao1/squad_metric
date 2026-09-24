"""Squadmetric: evidence-led squad and transfer analytics for FPL.

    python refresh.py        # or the Refresh data button, in the background
    streamlit run app.py

Layout only: every number comes from core.py, the lineup optimiser from
lineup.py, live AI from live_ai.py. Only the chosen section runs, so the AI is
called for Transfers or News only when that section is open. FPL_OFFLINE=1
for a demo that never touches the network. Live AI uses the viewer's own
OpenRouter key, held in session state only. Wide screens get the sidebar;
narrow ones a top bar with the same controls (CSS picks one).
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


# Controls drawn twice, in the sidebar (wide screens) and the top bar (narrow);
# CSS shows one. Each copy has its own widget key, synced to one canonical key.
VIEWS = ('wide', 'narrow')
SYNCED = {'entry_input': '', ai.KEY: '', 'live_ai': True}


def _init_synced():
    for name, default in SYNCED.items():
        st.session_state.setdefault(name, default)
        for view in VIEWS:
            st.session_state.setdefault(f'{name}_{view}', st.session_state[name])


def _sync(name, view):
    value = st.session_state[f'{name}_{view}']
    st.session_state[name] = value
    for other in VIEWS:
        st.session_state[f'{name}_{other}'] = value


def team_controls(view):
    st.text_input('FPL team ID', key=f'entry_input_{view}', placeholder='e.g. 3265946',
                  on_change=_sync, args=('entry_input', view))
    if st.button('Load team', type='primary', width='stretch', key=f'load_{view}'):
        st.session_state['entry'] = st.session_state[f'entry_input_{view}'].strip()


def ai_settings(view):
    """Bring your own OpenRouter key: session state only, never stored."""
    st.text_input('OpenRouter API key (optional)', type='password', key=f'{ai.KEY}_{view}',
                  placeholder='Paste your key', autocomplete='off',
                  on_change=_sync, args=(ai.KEY, view))
    st.caption('Enables live AI explanations. Your key stays in this browser '
               'session only. [Get a key](https://openrouter.ai/keys)')
    no_key = not ai.api_key()
    st.toggle('Live AI', disabled=core.OFFLINE or no_key, key=f'live_ai_{view}',
              on_change=_sync, args=('live_ai', view),
              help='Explanations and news ratings for your team. '
                   'Advisory only; never changes a suggestion.')
    if core.OFFLINE:
        st.caption('Offline mode: AI text is cached or template, as labelled.')
    elif no_key:
        st.caption('No key: AI text is cached or template, as labelled.')
    elif ai.rejected():
        st.warning(ai.REJECTED_NOTE)


def _start_refresh():
    started, message = refresher.start()
    st.session_state['refresh_toast'] = (
        message, ':material/sync:' if started else ':material/info:')


def refresh_button(view):
    st.button('Refresh data', key=f'refresh_{view}', width='stretch',
              disabled=core.OFFLINE, on_click=_start_refresh,
              help='Fetch the latest FPL data in the background. '
                   'The page keeps working meanwhile.')


def load():
    """(analysis or None, [(st method, text)] to show beside the team box)."""
    loaded = st.session_state.get('entry', '')
    if not loaded:
        return None, [('caption', 'Your team ID is in the URL of your FPL points page.')]
    if not loaded.isdigit():
        return None, [('error', 'A team ID is a number.')]
    try:
        t = core.team_analysis(loaded, core.OFFLINE)
    except Exception as e:
        log_failure(f'team {loaded} failed to load')
        return None, [('error', load_message(loaded, e))]
    notes = [('info', t['chip_note'])] if t.get('chip_note') else []
    if t['fell_back']:
        notes.append(('warning', f"Offline: showing cached data (team as of GW{t['team_gw']})."))
    return t, notes


def controls(gw):
    """The sidebar and the narrow-screen top bar. Returns the loaded team, or None."""
    if 'entry' not in st.session_state and st.query_params.get('team'):
        st.session_state['entry'] = st.query_params['team']
    _init_synced()
    # a changed key gets a fresh chance after a rejection; compared by hash
    seen = hash(ai.api_key() or '')
    if st.session_state.get('ai_key_hash') != seen:
        st.session_state['ai_key_hash'] = seen
        ai.budget().pop('rejected', None)

    with st.sidebar:
        s.section('Your team')
        team_controls('wide')
        s.section('AI explanations')
        ai_settings('wide')
        side = st.container()
        s.section('Data')
        refresh_button('wide')
    with st.container(key='mobilebar'):
        with st.container(horizontal=True, vertical_alignment='bottom', key='mobileteam'):
            team_controls('narrow')
        bar = st.container()
        with st.expander('AI settings'):
            ai_settings('narrow')
            refresh_button('narrow')

    t, notes = load()
    with side:
        if t:
            st.metric('Team value', s.money(t['r']['spend']))
            st.metric('Bank', s.money(t['r']['bank']))
        for kind, text in notes:
            getattr(st, kind)(text)
    with bar:
        st.markdown(s.tiles([('Gameweek', f'GW{gw}')] + (
            [('Bank', s.money(t['r']['bank'])), ('Team value', s.money(t['r']['spend']))]
            if t else [])), unsafe_allow_html=True)
        for kind, text in notes:
            getattr(st, kind)(text)
    return t


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


def render(name, t, pool):
    offline = core.OFFLINE
    if name in TEAM_TABS and t is None:
        s.card("<div class='sm-card-title'>Load a team to begin</div>"
               "<div class='sm-muted' style='margin-top:6px'>Enter your FPL team ID "
               "in the sidebar and press Load team.</div>")
    elif name == TABS[0]:
        team_tabs.my_team(t, pool, offline)
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

    gw, refreshed = core.data_stamp()
    s.header(gw, refreshed, core.OFFLINE)
    (_status_live if refresher.STATE['running'] else _status_idle)()
    t = controls(gw)
    name = section()
    try:
        render(name, t, pool)
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
