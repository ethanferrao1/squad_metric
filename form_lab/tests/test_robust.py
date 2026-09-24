"""Team loading fails with a clear reason, never a traceback."""
import socket
import urllib.error

import pytest

import app
import core
import fpl_api as api


def http(code):
    return urllib.error.HTTPError('url', code, 'x', {}, None)


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setattr(api, 'current_gw', lambda boot=None: 5)
    core.load_team.clear()


def test_an_unknown_id_is_a_404(monkeypatch, online):
    def picks(*a, **k):
        raise http(404)
    monkeypatch.setattr(api, 'load_team', picks)
    monkeypatch.setattr(api, '_get', picks)
    with pytest.raises(urllib.error.HTTPError) as e:
        core.load_team('987654321', False)
    assert 'No FPL team has ID' in app.load_message('987654321', e.value)


def test_a_new_team_without_picks_says_so(monkeypatch, online):
    def picks(*a, **k):
        raise http(404)
    monkeypatch.setattr(api, 'load_team', picks)
    monkeypatch.setattr(api, '_get', lambda path, refresh=False: {'id': 1})
    with pytest.raises(core.NoPicks) as e:
        core.load_team('987654322', False)
    assert 'no picks for GW5' in app.load_message('987654322', e.value)


def test_a_timeout_falls_back_to_the_cached_team(monkeypatch, online):
    calls = []

    def picks(entry, gw=None, refresh=False):
        calls.append(gw)
        if len(calls) == 1:
            raise socket.timeout('timed out')
        return [1, 2], 0
    monkeypatch.setattr(api, 'load_team', picks)
    monkeypatch.setattr(api, 'cached_event', lambda entry: 4)
    codes, bank, gw, fell_back = core.load_team('987654323', False)
    assert (codes, gw, fell_back) == ([1, 2], 4, True)


def test_a_timeout_with_nothing_cached_is_explained(monkeypatch, online):
    def picks(*a, **k):
        raise socket.timeout('timed out')
    monkeypatch.setattr(api, 'load_team', picks)
    monkeypatch.setattr(api, 'cached_event', lambda entry: None)
    with pytest.raises(FileNotFoundError) as e:
        core.load_team('987654324', False)
    assert 'not cached' in app.load_message('987654324', e.value)


def test_every_active_chip_gets_a_note():
    for chip in ('freehit', 'bboost', '3xc', 'wildcard'):
        assert 'GW7' in core.CHIP_NOTES[chip].format(gw=7)
