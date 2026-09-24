"""Check the app's data path for every cached team, or for given IDs live.

    python -m form_lab.app_check                  # OFFLINE=True, cached teams
    python -m form_lab.app_check --online 12 34   # live, through the same path

Runs what each tab computes: the Build squad, the rating and captain, and the
Improve tab's swaps. Offline, the network is blocked outright, so a cache
miss cannot hide.
"""
import sys
import urllib.request


def go_offline():
    """Make any network call raise."""
    def blocked(*a, **k):
        raise OSError('network disabled for the offline check')
    urllib.request.urlopen = blocked


def cached_teams():
    import re
    import fpl_api

    ids = set()
    for f in fpl_api.CACHE_DIR.glob('entry_*_event_*_picks.json'):
        m = re.match(r'entry_(\d+)_event_(\d+)_picks', f.stem)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def check(entry_id, pool):
    import core as app

    codes, bank, team_gw, fell_back = app.load_team(entry_id, app.OFFLINE)
    proj, gw, histories_fell_back = app.blended(pool, tuple(codes), app.OFFLINE)
    r = app.rate_now(pool, codes, bank, proj)

    names = pool.drop_duplicates('code').set_index('code')['web_name']
    swaps, below = ([], False) if proj is None else app.swap_rows(pool, r, proj)
    top = [f"{s['out']}->{s['in']} {s['gain_pts']:+.1f} {s['verdict']}"
           for _, s in (swaps.iterrows() if len(swaps) else [])]
    return {
        'entry': entry_id, 'team_gw': team_gw,
        'fell_back': fell_back or histories_fell_back,
        'rating': f"{r['pct']:.1f}%",
        'captain': names.get(r['captain'], r['captain']),
        'out_next_gw': ', '.join(names.get(c, str(c))
                                 for c in sorted(r['unavailable'])) or '-',
        'below_margin': below, 'swaps': ' | '.join(top),
        'benched_non_regulars': benched_non_regulars(swaps, r, proj, pool),
    }


def benched_non_regulars(swaps, r, proj, pool):
    """Suggested incoming players who would sit on the bench without being regulars."""
    import pandas as pd
    import rater as rt

    if not len(swaps):
        return 0
    by_name = pool.drop_duplicates('web_name').set_index('web_name')['code']
    ppg = proj.set_index('code')['blended_ppg']
    regular = set(proj.loc[proj['regular'], 'code'])
    squad = pool[pool['code'].isin(r['players']['code'])]

    bad = 0
    for _, s in swaps.iterrows():
        out_code, in_code = by_name[s['out']], by_name[s['in']]
        trial = pd.concat([squad[squad['code'] != out_code],
                           pool[pool['code'] == in_code]])
        trial = trial.assign(blended_ppg=trial['code'].map(ppg)).dropna(
            subset=['blended_ppg'])
        _, xi, _ = rt.squad_points(trial, 'blended_ppg')
        if in_code not in set(xi['code']) and in_code not in regular:
            bad += 1
    return bad


def main(entries=None):
    import pandas as pd
    import core as app
    import fpl_api
    from form_lab import data

    online = entries is not None
    app.OFFLINE = not online
    if not online:
        go_offline()

    pool = app.load_pool()
    squad, xi, captain = app.optimal_squad(pool, app.OFFLINE)
    names = pool.drop_duplicates('code').set_index('code')['web_name']
    bench = [c for c in squad['code'] if c not in set(xi['code'])]
    regulars = data.live_regulars(bench, fpl_api.current_gw())
    gw, refreshed = app.data_stamp()
    print(f'{"ONLINE" if online else "OFFLINE=True"} · data GW{gw}, '
          f'refreshed {refreshed}')
    print(f'Build tab: {len(squad)} players, spend '
          f'{squad["price"].sum() / 10:.1f}m, captain '
          f'{names.get(captain, captain)}; bench '
          f'{", ".join(names.get(c, str(c)) for c in bench)} '
          f'({sum(c in regulars for c in bench)}/4 regulars)\n')

    rows = []
    for entry in entries or cached_teams():
        try:
            rows.append(check(entry, pool))
        except Exception as e:
            rows.append({'entry': entry,
                         'rating': f'FAILED {type(e).__name__}: {e}'})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 120)
    print(pd.DataFrame(rows).to_string(index=False))
    if not online:
        print('\nno network was used')


if __name__ == '__main__':
    args = sys.argv[1:]
    main([int(a) for a in args[1:]] if args[:1] == ['--online'] else None)
