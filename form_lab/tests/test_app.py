"""The app renders every section for two cached teams offline, and players open their panel."""
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


def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def load(at, entry):
    at.sidebar.text_input[0].input(entry)
    at.sidebar.button[0].click()
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
        assert any(m.label == 'Team value' for m in at.sidebar.metric)
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
    assert any('not cached' in e.value for e in at.sidebar.error)


def test_a_non_numeric_team_id_is_explained(offline):
    at = app()
    load(at, 'abc')
    assert any('number' in e.value for e in at.sidebar.error)
