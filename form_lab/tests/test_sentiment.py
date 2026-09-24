"""Validation and parsing tests for the news judge, on hand-made inputs."""
import json

import pytest

from form_lab import sentiment as sm

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Palmer returns to training</title><source>BBC</source>
<pubDate>Mon, 22 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Chelsea boss praises Palmer</title><source>Sky</source>
<pubDate>Sun, 21 Sep 2026 09:00:00 GMT</pubDate></item>
</channel></rss>"""

GOOD = {'sentiment': 'positive', 'flags': ['role_change'], 'confidence': 'high',
        'summary': 'He is back in training and starting again.', 'evidence': [1]}


@pytest.fixture(autouse=True)
def no_roster(monkeypatch):
    """The other-player check would otherwise need the live bootstrap."""
    monkeypatch.setattr(sm, '_other_names', lambda subject: {'Haaland', 'Saka'})


def ok(payload, n=2, names=('Palmer',)):
    return sm.validate(json.dumps(payload), n, names)


def test_parses_titles_sources_and_dates():
    items = sm._parse_rss(RSS)
    assert len(items) == 2
    assert items[0]['title'] == 'Palmer returns to training'
    assert items[0]['source'] == 'BBC'
    assert '22 Sep 2026' in items[0]['date']


def test_caps_the_headline_count():
    assert len(sm._parse_rss(RSS, max_headlines=1)) == 1


def test_valid_verdict_passes():
    v, reason = ok(GOOD)
    assert reason == 'ok'
    assert v['sentiment'] == 'positive'
    assert v['evidence'] == [1]


def test_code_fences_are_stripped():
    fenced = f'```json\n{json.dumps(GOOD)}\n```'
    v, reason = sm.validate(fenced, 2, ('Palmer',))
    assert reason == 'ok'
    assert v['sentiment'] == 'positive'


def test_unparseable_json_falls_back_to_neutral():
    v, reason = sm.validate('not json at all', 2, ('Palmer',))
    assert v == sm.NEUTRAL
    assert reason == 'not JSON'


def test_unknown_sentiment_is_rejected():
    v, reason = ok({**GOOD, 'sentiment': 'bullish'})
    assert v == sm.NEUTRAL
    assert 'sentiment' in reason


def test_unknown_flag_is_rejected():
    v, reason = ok({**GOOD, 'flags': ['vibes']})
    assert v == sm.NEUTRAL
    assert 'flags' in reason


def test_unknown_confidence_is_rejected():
    v, reason = ok({**GOOD, 'confidence': 'certain'})
    assert v == sm.NEUTRAL


def test_evidence_beyond_the_headline_count_is_rejected():
    v, reason = ok({**GOOD, 'evidence': [3]}, n=2)
    assert v == sm.NEUTRAL
    assert 'headline number' in reason


def test_non_integer_evidence_is_rejected():
    v, reason = ok({**GOOD, 'evidence': ['one']})
    assert v == sm.NEUTRAL


def test_empty_evidence_is_allowed():
    v, reason = ok({**GOOD, 'evidence': []})
    assert reason == 'ok'


def test_summary_naming_another_player_is_rejected():
    v, reason = ok({**GOOD, 'summary': 'He outscored Haaland last week.'})
    assert v == sm.NEUTRAL
    assert 'Haaland' in reason


def test_summary_may_name_the_subject():
    v, reason = ok({**GOOD, 'summary': 'Palmer is back in training.'})
    assert reason == 'ok'


def test_no_headlines_gives_the_neutral_verdict():
    v, reason = sm.judge([], ('Palmer',))
    assert v == sm.NEUTRAL
    assert reason == 'no headlines'


def test_judge_falls_back_when_the_call_fails(monkeypatch):
    monkeypatch.setattr(sm, '_chat', lambda *a, **k: (None, 'rate limited (429)'))
    v, reason = sm.judge([{'title': 't', 'source': 's', 'date': 'd'}],
                         ('Palmer',))
    assert v == sm.NEUTRAL
    assert '429' in reason
    assert reason.startswith(sm.UNAVAILABLE)


def test_a_429_is_not_cached_as_a_verdict(monkeypatch, tmp_path):
    """Otherwise the fallback sticks and the model is never asked again."""
    monkeypatch.setattr(sm, 'VERDICT_DIR', tmp_path)
    monkeypatch.setattr(sm, 'headlines',
                        lambda c, **k: [{'title': 't', 'source': 's', 'date': 'd'}])
    monkeypatch.setattr(sm, '_player', lambda c: ('Cole Palmer', 'Chelsea', 'Palmer'))
    monkeypatch.setattr(sm, '_chat', lambda *a, **k: (None, 'rate limited (429)'))

    assert sm.assess(1) == sm.NEUTRAL
    assert list(tmp_path.glob('*.json')) == []


def test_a_real_verdict_is_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, 'VERDICT_DIR', tmp_path)
    monkeypatch.setattr(sm, 'headlines',
                        lambda c, **k: [{'title': 't', 'source': 's', 'date': 'd'}])
    monkeypatch.setattr(sm, '_player', lambda c: ('Cole Palmer', 'Chelsea', 'Palmer'))
    monkeypatch.setattr(sm, '_chat',
                        lambda *a, **k: (json.dumps({**GOOD, 'evidence': [1]}), None))

    assert sm.assess(1)['sentiment'] == 'positive'
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_no_headlines_is_cached(monkeypatch, tmp_path):
    """A genuinely empty result is stable, so it should not refetch forever."""
    monkeypatch.setattr(sm, 'VERDICT_DIR', tmp_path)
    monkeypatch.setattr(sm, 'headlines', lambda c, **k: [])
    monkeypatch.setattr(sm, '_player', lambda c: ('Cole Palmer', 'Chelsea', 'Palmer'))

    assert sm.assess(1) == sm.NEUTRAL
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_judge_validates_a_real_looking_reply(monkeypatch):
    monkeypatch.setattr(sm, '_chat', lambda *a, **k: (json.dumps(GOOD), None))
    v, reason = sm.judge([{'title': 't', 'source': 's', 'date': 'd'}],
                         ('Palmer',))
    assert reason == 'ok'
    assert v['sentiment'] == 'positive'


def test_headline_naming_the_player_is_kept():
    assert sm.mentions_player('Palmer returns to training',
                              'Cole Palmer', 'Palmer')


def test_club_roundup_without_the_name_is_dropped():
    assert not sm.mentions_player(
        'Chelsea transfers, latest news and gossip: live updates',
        'Cole Palmer', 'Palmer')


def test_first_name_alone_does_not_count_as_a_mention():
    assert not sm.mentions_player('Cole is back in the squad',
                                  'Cole Palmer', 'Palmer')


def test_hyphenated_surname_matches_either_part():
    assert sm.mentions_player('Calvert-Lewin scores again',
                              'Dominic Calvert-Lewin', 'Calvert-Lewin')
    assert sm.mentions_player('Lewin nets a brace',
                              'Dominic Calvert-Lewin', 'Calvert-Lewin')


def test_substring_of_a_longer_word_is_not_a_mention():
    assert not sm.mentions_player('Palmerston Park hosts the tie',
                                  'Cole Palmer', 'Palmer')


def test_headlines_are_numbered_from_one():
    text = sm._numbered([{'title': 'A', 'source': 'BBC', 'date': 'd'},
                         {'title': 'B', 'source': 'Sky', 'date': 'd'}])
    assert text.startswith('1. A (BBC')
    assert '\n2. B (Sky' in text


def test_neutral_verdict_is_never_mutated():
    """validate() hands back copies, so a caller cannot poison the fallback."""
    v, _ = sm.validate('nonsense', 1, ('Palmer',))
    v['sentiment'] = 'positive'
    assert sm.NEUTRAL['sentiment'] == 'neutral'
