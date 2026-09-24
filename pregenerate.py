"""Fill the explanation cache for every team already in data/cache/.

    python pregenerate.py
    python pregenerate.py --paid     # bills lx.PAID_MODEL instead

Run it once with a key and a connection; afterwards the app and the demo
read data/cache/explanations.json and need neither. Free OpenRouter models
are rate limited, so this pauses between calls and stops asking after a
string of refusals -- llm_explain already falls down a list of free models
and retries each 429 with backoff before a call counts as failed here.

--paid uses one cheap general instruct model (~$0.08 per million output
tokens) with no free fallbacks, for when the free tier's daily quota is
spent. A full run is a fraction of a cent.
"""
import argparse
import re
import time

import core as app
import build_live as bl
import fpl_api as api
import llm_explain as lx
import rater as rt

PAUSE = 4                # seconds between calls, to stay under the free tier
GIVE_UP_AFTER = 10       # consecutive failures before stopping


def cached_entry_ids():
    """Entry ids that already have picks cached, so this needs no lookups."""
    ids = set()
    for f in lx.CACHE.parent.glob('entry_*_event_*_picks.json'):
        m = re.match(r'entry_(\d+)_event_(\d+)_picks', f.stem)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def main(entry_ids=None, pause=PAUSE, paid=False):
    pool = bl.load()
    gw = api.current_gw()
    entry_ids = entry_ids or cached_entry_ids()
    print(f'gw {gw}, {len(entry_ids)} cached teams: {entry_ids}')

    model = lx.PAID_MODEL if paid else None
    if paid:
        print(f'--paid: billing {model}, free fallbacks off')
    else:
        print(f'free: {lx.MODEL}, falling back to '
              f'{", ".join(lx.FALLBACKS)}')

    made = hit = failed = 0
    by_model = {}
    streak = 0
    for entry in entry_ids:
        codes, bank = api.load_team(entry)
        r = rt.rate_squad(codes, pool, bank=bank)
        s = rt.suggest(r, pool, budget=r['budget'])
        print(f'\nentry {entry}: {len(s)} suggestions')

        for _, row in s.iterrows():
            key = lx.cache_key(gw, row['out'], row['in'])
            cached = lx._load_cache().get(key)
            if cached is not None:
                hit += 1
                was = cached.get('model') or 'unknown model'
                print(f'  cached   [{was}] {row["out"]} -> {row["in"]}')
                continue

            text, source, detail = app.explain(row, pool, use_llm=True,
                                               model=model)
            if source == 'llm':
                made += 1
                streak = 0
                by_model[detail] = by_model.get(detail, 0) + 1
                print(f'  llm      [{detail}] {row["out"]} -> '
                      f'{row["in"]}: {text}')
            else:
                failed += 1
                streak += 1
                # `detail` is why: a 429 that outlasted the retries reads
                # differently from a reply that failed the fact check
                print(f'  template {row["out"]} -> {row["in"]}  ({detail})')
                if streak >= GIVE_UP_AFTER:
                    print(f'\n{streak} failures in a row -- stopping. '
                          f'Check OPENROUTER_API_KEY (with ALLOW_ENV_KEY=1) and the rate limit, '
                          f'or rerun with --paid.')
                    return made, hit, failed
            time.sleep(pause)

    print(f'\nnew {made}, already cached {hit}, fell back {failed}')
    for m, n in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f'  {n:3d} from {m}')
    return made, hit, failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--paid', action='store_true',
                    help=f'bill {lx.PAID_MODEL} instead of the free models')
    ap.add_argument('--pause', type=float, default=PAUSE,
                    help='seconds between calls (default %(default)s)')
    args = ap.parse_args()
    main(pause=args.pause, paid=args.paid)
