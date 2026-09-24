"""The app renders every section for two cached teams offline, players open their
panel, and the My Team setup folds away once the team and key are settled."""
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEAMS = ('3265946', '1')              # one with an injured starter, one without
SECTIONS = ['My Team', 'Lineup', 'Transfers', 'Players', 'News',
            'Build']


@pytest.fixture
def offline(monkeypatch):
    def blocked(*a, **k):
        raise OSError('network disabled in tests')
    monkeypatch.setattr(urllib.request, 'urlopen', blocked)
    monkeypatch.setenv('FPL_OFFLINE', '1')
    import core
    monkeypatch.setattr(core, 'OFFLINE', True)


@pytest.fixture
def online(monkeypatch):
    """The deployed setup: AI possible, keys checked by a stub, no network."""
    def blocked(*a, **k):
        raise OSError('network disabled in tests')
    monkeypatch.setattr(urllib.request, 'urlopen', blocked)
    monkeypatch.delenv('FPL_OFFLINE', raising=False)
    import core
    import llm_explain
    monkeypatch.setattr(core, 'OFFLINE', False)
    monkeypatch.setattr(llm_explain, 'check_key', lambda key: key == 'good-key')


def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def setup_shown(at):
    return any(w.key == 'setup_team' for w in at.text_input)


def load(at, entry):
    """Enter a team ID in the My Team page's setup form, reopening it if folded."""
    if at.radio(key='section').value != 'My Team':
        show(at, 'My Team')
    if not setup_shown(at):
        at.button(key='setup_change').click()
        at.run()
    at.text_input(key='setup_team').input(entry)
    at.button(key='setup_load').click()
    at.run()
    assert not at.exception, at.exception


def submit(at, entry, key='', button='setup_load'):
    at.text_input(key='setup_team').input(entry)
    at.text_input(key='setup_key').input(key)
    at.button(key=button).click()
    at.run()
    assert not at.exception, at.exception


def show(at, name):
    at.radio(key='section').set_value(name)
    at.run()
    assert not at.exception, at.exception


def test_every_section_renders_offline(offline):
    at = app()
    assert list(at.radio(key='section').options) == SECTIONS
    ratings = {}
    for entry in TEAMS:
        load(at, entry)
        assert any('Team value' in m.value for m in at.markdown)
        for name in SECTIONS:
            show(at, name)
            assert not at.error, (name, [e.value for e in at.error])
            assert len(at.main.children) > 2, f'{name} is empty'
            assert not any('could not be shown' in m.value for m in at.markdown), name
        show(at, SECTIONS[0])
        ratings[entry] = at.metric[0].value
    # loading a second team must actually replace the first
    assert len(set(ratings.values())) == len(TEAMS), ratings


def _open(at, key, name):
    """Click a player's name chip; return whether his panel opened."""
    chips = [g for g in at.get('button_group') if g.key == key]
    assert chips, f'no chips called {key}'
    chips[0].set_value([name])
    at.run()
    assert not at.exception, at.exception
    return any('avatar-lg' in m.value and f'>{name}</div>' in m.value
               for m in at.markdown)


def test_clicking_a_player_in_each_section_opens_his_panel(offline):
    at = app()
    load(at, '3265946')
    for section, key, name in ((SECTIONS[0], 'team-pick', 'Haaland'),
                               (SECTIONS[1], 'lineup-changes-pick', 'Alderete'),
                               (SECTIONS[2], 'swapc0-pick', 'Bogle'),
                               (SECTIONS[2], 'plan1-pick', 'Wieffer'),
                               (SECTIONS[5], 'wc-pick', 'Haaland')):
        show(at, section)
        assert _open(at, key, name), f'{key}: no panel for {name}'


def test_an_unknown_team_gives_a_message_not_a_traceback(offline):
    at = app()
    load(at, '999999999')
    assert not at.exception
    assert any('not cached' in e.value for e in at.error)


def test_a_non_numeric_team_id_is_explained(offline):
    at = app()
    load(at, 'abc')
    assert any('number' in e.value for e in at.error)


def test_setup_folds_away_once_the_team_and_key_are_valid(online):
    at = app()
    assert setup_shown(at)
    submit(at, '3265946')                         # a team but no key yet: stays
    assert setup_shown(at) and any('Team value' in m.value for m in at.markdown)
    submit(at, '3265946', 'bad-key')              # a rejected key: stays, says why
    assert setup_shown(at) and any('rejected' in e.value for e in at.error)
    submit(at, '3265946', 'good-key')
    assert not setup_shown(at)
    assert at.session_state['openrouter_key'] == 'good-key'
    assert any(t.key == 'live_ai_toggle' for t in at.toggle)
    at.button(key='setup_change').click()         # and back again
    at.run()
    assert setup_shown(at)


def test_continuing_without_ai_folds_the_setup_too(online):
    at = app()
    submit(at, '3265946', button='setup_skip')
    assert not setup_shown(at)


def test_an_invalid_team_keeps_the_setup_open(online):
    at = app()
    submit(at, '999999999', 'good-key')
    assert setup_shown(at) and any('not cached' in e.value for e in at.error)
