from pathlib import Path
import pandas as pd

# Anchored to this file, not the working directory, so the pipeline loads the
# same CSVs whether it's imported from the project root or from notebooks/.
DATA_DIR = Path(__file__).resolve().parent / 'data'

# In progress: in SEASONS (build_season_panel/club_tables need its roster,
# prices and clubs) but never a feature or target season -- it has no full
# season of gameweek data, so the season-long aggregates would be garbage.
CURRENT_SEASON = '26_27'

SEASONS = ['16_17', '17_18', '18_19', '19_20', '20_21',
           '21_22', '22_23', '23_24', '24_25', '25_26', '26_27']
BAD_FEATURE_SEASONS = {'21_22'}

FEATURE_SEASONS = [s for s in SEASONS
                   if s not in BAD_FEATURE_SEASONS and s != CURRENT_SEASON]

def _read_csv(path):
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(path, encoding = enc, low_memory =False)
        except UnicodeDecodeError:
            continue
    raise IOError(f'could not decode {path}')

def read_gw(season):
    return _read_csv(DATA_DIR / f'merged_gw_{season}.csv')

def read_players(season):
    return _read_csv(DATA_DIR / f'players_raw_20{season}.csv')

def season_integrity(season):
    gw = read_gw(season)
    players = read_players(season)

    surname  = players.set_index('id')['second_name'].str.lower()
    expected = gw['element'].map(surname)
    actual   = gw['name'].str.lower().str.replace('_', ' ')

    ok = [isinstance(e, str) and isinstance(a, str) and e in a
          for e, a in zip(expected, actual)]

    return sum(ok) / len(ok)
def check_all():
    return pd.Series({s: season_integrity(s) for s in SEASONS})