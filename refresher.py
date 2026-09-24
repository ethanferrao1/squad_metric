"""The app's Refresh data button: refresh.run in a background thread.

One refresh per server at a time (a global lock), a cooldown after each
success, and never automatic. STATE is shared by every session; each session
compares STATE['gen'] with the generation it last saw to know a refresh ended.
"""
import logging
import math
import threading
import time

import refresh

COOLDOWN = 600                     # seconds after a successful refresh
_lock = threading.Lock()
STATE = {'running': False, 'phase': '', 'done': None, 'total': None, 'gen': 0,
         'ok': None, 'gw': None, 'last_ok': 0.0}


def reset():
    """Back to never-refreshed (tests)."""
    STATE.update(running=False, phase='', done=None, total=None, gen=0, ok=None,
                 gw=None, last_ok=0.0)


def clear_caches():
    """Drop the st.cache_data loaders that read the refreshed files."""
    import core
    for fn in (core.load_pool, core.load_team, core.blended, core.availability_info,
               core.optimal_squad, core.team_analysis, core.transfer_plans,
               core.wildcard_now, core.free_hit_now, core.build_views, core.players):
        fn.clear()


def status():
    """'Refreshing: fetching players 240/667…', or '' when idle."""
    if not STATE['running']:
        return ''
    count = f" {STATE['done']}/{STATE['total']}" if STATE['total'] else ''
    return f"Refreshing: {STATE['phase']}{count}…"


def _progress(phase, done=None, total=None):
    STATE.update(phase=phase, done=done, total=total)


def _work(run):
    try:
        gw = run(progress=_progress)
        clear_caches()
        STATE.update(ok=True, gw=gw, last_ok=time.time())
    except Exception as e:
        logging.error('refresh failed, old data kept: %s: %s', type(e).__name__, e)
        STATE.update(ok=False)
    finally:
        STATE.update(running=False, phase='')
        STATE['gen'] += 1
        _lock.release()


def start(run=None):
    """(started, message). Starts a background refresh unless one is running
    or the last one finished under COOLDOWN seconds ago."""
    if not _lock.acquire(blocking=False):
        return False, 'A data refresh is already running.'
    wait = STATE['last_ok'] + COOLDOWN - time.time()
    if wait > 0:
        _lock.release()
        return False, (f'Data was refreshed recently. Try again in '
                       f'{math.ceil(wait / 60)} min.')
    global _thread
    STATE.update(running=True, phase='starting', done=None, total=None, ok=None)
    _thread = threading.Thread(target=_work, args=(run or refresh.run,), daemon=True,
                               name='squadmetric-refresh')
    _thread.start()
    return True, 'Refreshing data in the background.'


_thread = None


def wait(timeout=None):
    """Block until the running refresh ends (tests and scripts)."""
    if _thread is not None:
        _thread.join(timeout)
