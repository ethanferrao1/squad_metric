"""Time the live AI for given teams through the app's own job path.

    python -m form_lab.ai_check 123 456             # free models, no cost
    python -m form_lab.ai_check --paid 123 456      # PAID_MODEL, as the app uses
    python -m form_lab.ai_check --offline 123 456   # network blocked: fallbacks only

Reports time to the first explanation, time to all results, calls and cost
(from data/cache/llm_log.csv).
"""
import os
import sys
import time
import urllib.request

import pandas as pd


def run(entry, offline):
    import core
    import live_ai
    import llm_explain as lx
    t = core.team_analysis(str(entry), offline)
    free = core.free_transfers(int(entry), t['gw']) or 1
    plans, _ = core.transfer_plans(str(entry), offline, free, 0)
    pool = core.load_pool()
    explain = core.explain_jobs(t, pool, plans)
    news = core.news_jobs(core.news_targets(t, plans), t['gw'])
    start, first, labels = time.time(), {}, []

    def land(key, result):
        kind = 'news' if '|news|' in key else 'explain'
        first.setdefault(kind, time.time() - start)
        labels.append(result['label'])

    before = len(pd.read_csv(live_ai.LOG)) if live_ai.LOG.exists() else 0
    live_ai.resolve(explain + news, land, {'used': 0}, t['gw'], live=not offline,
                    api_key=lx.env_key())
    log = pd.read_csv(live_ai.LOG).iloc[before:] if live_ai.LOG.exists() else pd.DataFrame()
    return {'entry': entry, 'rating': f"{t['r']['pct']:.1f}%",
            'jobs': len(explain) + len(news),
            'first explanation (s)': round(first.get('explain', float('nan')), 1),
            'all results (s)': round(time.time() - start, 1),
            'calls': len(log), 'cost ($)': round(float(log['cost_usd'].sum()), 5) if len(log) else 0.0,
            'labels': {l: labels.count(l) for l in sorted(set(labels))}}


def main(args):
    offline = '--offline' in args
    if '--paid' not in args:
        os.environ['SQUADMETRIC_FREE_AI'] = '1'
    if offline:
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError('offline'))
    rows = [run(int(a), offline) for a in args if a.isdigit()]
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1:])
