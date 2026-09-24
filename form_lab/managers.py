"""Real-manager benchmark: sample FPL entries and keep their season totals.

    python -m form_lab.managers

Entry ids are drawn uniformly from 1..MAX_ENTRY and their /history/ "past"
rows kept for the simulated seasons. Cached, so this runs once.
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from form_lab import data

MAX_ENTRY = 12_000_000
TARGET = 400
PAUSE = 1.0
TIMEOUT = 20
UA = 'Mozilla/5.0'
API = 'https://fantasy.premierleague.com/api'
DIR = data.CACHE / 'managers'

# "24_25" -> the season_name FPL uses in a history "past" row
SEASON_NAME = {'23_24': '2023/24', '24_25': '2024/25', '25_26': '2025/26'}


def _path(season):
    return DIR / f'{season}.csv'


def _fetch(entry_id):
    """One entry's history, or None if it does not exist."""
    url = f'{API}/entry/{entry_id}/history/'
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return None if e.code in (404, 403) else None
    except Exception:
        return None


def _past_rows(history, seasons):
    """season -> (total_points, rank) for the seasons this entry played."""
    out = {}
    for row in (history or {}).get('past', []):
        for season, name in SEASON_NAME.items():
            if season in seasons and row.get('season_name') == name:
                out[season] = (row.get('total_points'), row.get('rank'))
    return out


def collect(seasons=tuple(SEASON_NAME), target=TARGET, pause=PAUSE,
            seed=None, max_tries=None):
    """Sample entries until each season has `target` managers. Resumable."""
    DIR.mkdir(parents=True, exist_ok=True)
    rows = {s: _read(s) for s in seasons}
    seen = {int(i) for s in seasons for i in rows[s].get('entry_id', [])}
    rng = random.Random(seed)

    tries = 0
    max_tries = max_tries or target * 20
    while any(len(rows[s]) < target for s in seasons) and tries < max_tries:
        tries += 1
        entry = rng.randint(1, MAX_ENTRY)
        if entry in seen:
            continue
        seen.add(entry)

        past = _past_rows(_fetch(entry), seasons)
        time.sleep(pause)
        for season, (total, rank) in past.items():
            if len(rows[season]) >= target or total is None:
                continue
            rows[season] = pd.concat([rows[season], pd.DataFrame([{
                'entry_id': entry, 'total_points': total, 'rank': rank}])],
                ignore_index=True)

        if tries % 25 == 0:
            got = ', '.join(f'{s}={len(rows[s])}' for s in seasons)
            print(f'  {tries} ids tried: {got}', flush=True)
            for s in seasons:
                _write(s, rows[s])

    for s in seasons:
        _write(s, rows[s])
        print(f'{s}: {len(rows[s])} managers from {tries} ids')
    return rows


def _read(season):
    p = _path(season)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=['entry_id', 'total_points', 'rank'])


def _write(season, frame):
    frame.to_csv(_path(season), index=False)


def load(season):
    """The cached sample for one season."""
    p = _path(season)
    if not p.exists() or not len(_read(season)):
        raise FileNotFoundError(f'no manager sample at {p}')
    d = _read(season).dropna(subset=['total_points'])
    d['rank_percentage'] = d['rank'].rank(pct=True) * 100
    return d.sort_values('total_points').reset_index(drop=True)


def summary(sample):
    """Mean, median and quartiles of the sampled season totals."""
    t = sample['total_points']
    return {'n': len(t), 'mean': float(t.mean()), 'median': float(t.median()),
            'p25': float(t.quantile(0.25)), 'p75': float(t.quantile(0.75))}


def top_pct(sample, total):
    """Percentage of managers a score of `total` would have beaten to, i.e.
    'finished in the top X%'."""
    beaten = (sample['total_points'] < total).mean()
    return round(100 * (1 - beaten), 1)


def estimated_rank(sample, total):
    """Overall rank implied by interpolating total against rank."""
    d = sample.dropna(subset=['rank']).sort_values('total_points')
    if len(d) < 2:
        return np.nan
    return float(np.interp(total, d['total_points'], d['rank'],
                           left=d['rank'].iloc[0], right=d['rank'].iloc[-1]))


def report(seasons=tuple(SEASON_NAME)):
    rows = []
    for s in seasons:
        try:
            sample = load(s)
        except FileNotFoundError:
            print(f'{s}: no sample yet')
            continue
        rows.append({'season': s, **summary(sample)})
    if rows:
        print(pd.DataFrame(rows).round(1).to_string(index=False))
    print('\nreal managers used chips and took hits; the simulation did '
          'neither. Random ids include abandoned teams, so read the median '
          'and the rank, not the mean.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--target', type=int, default=TARGET)
    ap.add_argument('--pause', type=float, default=PAUSE)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    if args.report:
        report()
    else:
        collect(target=args.target, pause=args.pause, seed=args.seed)
        report()
