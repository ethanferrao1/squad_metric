"""Pre-fill the live-AI caches for every cached team: the offline backup for a demo.

    python warmup.py

Builds the same explanation and news-rating jobs the app would, skips any
already cached for this gameweek, prints the call count and estimated cost,
and asks before making a single paid call. Capped at live_ai.MAX_CALLS a run.
Local only: needs OPENROUTER_API_KEY in the environment or .env AND ALLOW_ENV_KEY=1.
"""
import time

import core
import live_ai
import llm_explain as lx
from form_lab import app_check


def jobs_for(entry):
    """Every live-AI job the app would run for one cached team."""
    t = core.team_analysis(str(entry), True)
    if t['proj'] is None:
        return [], None
    free = core.free_transfers(entry, t['gw']) or 1
    plans, _ = core.transfer_plans(str(entry), True, free, 0)
    pool = core.load_pool()
    return (core.explain_jobs(t, pool, plans)
            + core.news_jobs(core.news_targets(t, plans), t['gw'])), t['gw']


def main():
    jobs, gw = {}, None
    for entry in app_check.cached_teams():
        found, gw = jobs_for(entry)
        jobs.update({j['key']: j for j in found})
    todo = [j for j in jobs.values() if not live_ai._cached(j, gw)]
    calls, dollars = live_ai.estimate(todo[:live_ai.MAX_CALLS])
    model = live_ai.model_kw()['model']
    print(f'GW{gw}: {len(jobs)} jobs across {len(app_check.cached_teams())} teams, '
          f'{len(todo)} not yet cached')
    print(f'this run: {calls} calls on {model}, estimated ${dollars:.4f}'
          + (f' (capped at {live_ai.MAX_CALLS}; run again for the rest)'
             if len(todo) > live_ai.MAX_CALLS else ''))
    print(f"API key present: {'yes' if lx.env_key() else 'no (set OPENROUTER_API_KEY and ALLOW_ENV_KEY=1)'}")
    if not todo or input('Proceed? [y/N] ').strip().lower() != 'y':
        print('nothing called')
        return

    start, outcomes = time.time(), {}
    live_ai.resolve(todo[:live_ai.MAX_CALLS],
                    lambda k, r: outcomes.__setitem__(k, r['label']),
                    {'used': 0}, gw, live=True, api_key=lx.env_key())
    counts = {lab: list(outcomes.values()).count(lab) for lab in set(outcomes.values())}
    print(f'done in {time.time() - start:.0f}s: {counts}. Costs are in {live_ai.LOG}')


if __name__ == '__main__':
    main()
