"""Availability over the next five gameweeks, from FPL's status flags.

    status i/u/n, chance 0 or null -> out for all 5 gameweeks
    status s (suspended)           -> out next gameweek, normal after
    status d (doubtful)            -> next gameweek x chance/100, normal after
    status a                       -> unchanged

An i/u/n flag with a non-zero chance is treated as doubtful. The app uses
this; the backtest has no injury history and does not.
"""
from form_lab.form import SEASON_GWS, SWAP_HORIZON as WINDOW
OUT = ('i', 'u', 'n')
SUSPENDED = 's'
DOUBTFUL = 'd'
DOUBTFUL_DEFAULT = 50          # a 'd' flag that carries no chance figure


def multipliers(status, chance, window=WINDOW):
    """Per-gameweek availability, next gameweek first."""
    if status in OUT and not chance:
        return [0.0] * window
    if status == SUSPENDED:
        return [0.0] + [1.0] * (window - 1)
    if status == DOUBTFUL or status in OUT:
        pct = DOUBTFUL_DEFAULT if chance is None else chance
        return [pct / 100] + [1.0] * (window - 1)
    return [1.0] * window


def lookup(elements):
    """code -> {status, chance, news, mult} from bootstrap elements."""
    return {e['code']: {'status': e.get('status', 'a'),
                        'chance': e.get('chance_of_playing_next_round'),
                        'news': (e.get('news') or '').strip(),
                        'mult': multipliers(e.get('status', 'a'),
                                            e.get('chance_of_playing_next_round'))}
            for e in elements}


def out_next(info):
    """True if he cannot play next gameweek."""
    return info['mult'][0] == 0


def season_factor(info, season_gws=SEASON_GWS):
    """Share of a season projection left after the gameweeks he will miss."""
    return 1 - (len(info['mult']) - sum(info['mult'])) / season_gws
