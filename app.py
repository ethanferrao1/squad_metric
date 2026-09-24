"""Squadmetric: evidence-led squad and transfer analytics for FPL.

    python refresh.py        # before a demo: pool, prices, histories, images
    streamlit run app.py

Layout only: every number comes from core.py, the lineup optimiser from
lineup.py, live AI from live_ai.py. Only the chosen section runs, so the AI is
called for Transfers or News only when that section is open. FPL_OFFLINE=1
for a demo that never touches the network. Live AI uses the viewer's own
OpenRouter key from the sidebar, held in session state only.
"""
import logging
import traceback

import streamlit as st

import core
import llm_explain
from ui import ai, build, pool as pool_tabs, style as s, team as team_tabs

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


def ai_settings():
    """Bring your own OpenRouter key: session state only, never stored."""
    st.text_input('OpenRouter API key (optional)', type='password', key=ai.KEY,
                  placeholder='Paste your key', autocomplete='off')
    st.caption('Enables live AI explanations. Your key stays in this browser '
               'session only. [Get a key](https://openrouter.ai/keys)')
    # a changed key gets a fresh chance after a rejection; compared by hash
    seen = hash(ai.api_key() or '')
    if st.session_state.get('ai_key_hash') != seen:
        st.session_state['ai_key_hash'] = seen
        ai.budget().pop('rejected', None)

    no_key = not ai.api_key()
    st.toggle('Live AI', value=True, disabled=core.OFFLINE or no_key,
              key='live_ai', help='Explanations and news ratings for your team. '
                                  'Advisory only; never changes a suggestion.')
    if core.OFFLINE:
        st.caption('Offline mode: AI text is cached or template, as labelled.')
    elif no_key:
        st.caption('No key: AI text is cached or template, as labelled.')
    elif ai.rejected():
        st.warning(ai.REJECTED_NOTE)


def sidebar():
    """Team ID box and switches. Returns the loaded team's analysis, or None."""
    if 'entry' not in st.session_state and st.query_params.get('team'):
        st.session_state['entry'] = st.query_params['team']
    with st.sidebar:
        s.section('Your team')
        entry = st.text_input('FPL team ID', key='entry_input',
                              placeholder='e.g. 3265946')
        if st.button('Load team', type='primary', width='stretch'):
            st.session_state['entry'] = entry.strip()
        s.section('AI explanations')
        ai_settings()

        loaded = st.session_state.get('entry', '')
        if not loaded:
            st.caption('Your team ID is in the URL of your FPL points page.')
            return None
        if not loaded.isdigit():
            st.error('A team ID is a number.')
            return None
        try:
            t = core.team_analysis(loaded, core.OFFLINE)
        except Exception as e:
            log_failure(f'team {loaded} failed to load')
            st.error(load_message(loaded, e))
            return None
        st.metric('Team value', s.money(t['r']['spend']))
        st.metric('Bank', s.money(t['r']['bank']))
        if t.get('chip_note'):
            st.info(t['chip_note'])
        if t['fell_back']:
            st.warning(f"Offline: showing cached data (team as of GW{t['team_gw']}).")
        return t


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

    gw, refreshed = core.data_stamp()
    s.header(gw, refreshed, core.OFFLINE)
    t = sidebar()
    name = section()
    try:
        render(name, t, pool)
    except Exception:
        log_failure(f'section {name} failed')
        s.card("<div class='sm-card-title'>This section could not be shown</div>"
               "<div class='sm-muted' style='margin-top:6px'>Something in the data for "
               "it is missing or unexpected. The other sections still work; "
               "try Load team again or run refresh.py.</div>")


if __name__ == '__main__':
    main()
