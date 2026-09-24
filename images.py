"""Player photos, club badges and shirts: fetched from the Premier League at
runtime the first time each is shown, cached in data/cache/img/ (gitignored,
never committed), served as data URIs, with a neutral placeholder when one
cannot be fetched.

    fetch_all(pause)     # optional: refresh.py pre-fills the whole cache

Runtime fetches use a short timeout, skip any URL that already failed, and
stop for FAIL_COOLDOWN seconds after FAIL_LIMIT failures in a row, so a
blocked network costs one quick failure rather than one per image.
FPL_OFFLINE=1 never fetches.
"""
import base64
import os
import threading
import time
import urllib.request

import fpl_api as api

DIR = api.CACHE_DIR / 'img'
UA = {'User-Agent': 'Mozilla/5.0'}
TIMEOUT = 20
RUNTIME_TIMEOUT = 4
FAIL_LIMIT, FAIL_COOLDOWN = 3, 300
PAUSE = 0.3

_lock = threading.Lock()
_state = {'failed': set(), 'streak': 0, 'paused_until': 0.0}
_uris = {}

PHOTO = ('https://resources.premierleague.com/premierleague/photos/players/'
         '110x140/p{id}.png')
BADGE = 'https://resources.premierleague.com/premierleague/badges/70/t{team}.png'
SHIRT = ('https://fantasy.premierleague.com/dist/img/shirts/standard/'
         'shirt_{team}{keeper}-66.png')

PLACEHOLDER = {
    'photo': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 110 140">'
              '<rect width="110" height="140" fill="#e2e8f0"/>'
              '<circle cx="55" cy="52" r="24" fill="#94a3b8"/>'
              '<path d="M15 140c0-30 18-48 40-48s40 18 40 48z" fill="#94a3b8"/>'
              '</svg>'),
    'badge': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 70">'
              '<path d="M35 4l27 10v20c0 17-12 28-27 32C20 62 8 51 8 34V14z" '
              'fill="#cbd5e1"/></svg>'),
    'shirt': ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 66 87">'
              '<path d="M18 4l-16 12 8 14 7-4v57h32V26l7 4 8-14-16-12'
              'c-3 6-9 9-15 9s-12-3-15-9z" fill="#94a3b8"/></svg>'),
}


def photo_path(photo_field):
    """bootstrap `photo` '12345.jpg' -> its cache file."""
    return DIR / f"photo_{str(photo_field).split('.')[0]}.png"


def badge_path(team_code):
    return DIR / f'badge_{team_code}.png'


def shirt_path(team_code, keeper=False):
    return DIR / f"shirt_{team_code}{'_1' if keeper else ''}.png"


def _download(url, path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        path.write_bytes(r.read())


def wanted(boot=None):
    """[(url, path)] for every photo, badge and shirt the app can show."""
    boot = boot or api.bootstrap()
    teams = {t['id']: t['code'] for t in boot['teams']}
    out = []
    for code in sorted(set(teams.values())):
        out += [(BADGE.format(team=code), badge_path(code)),
                (SHIRT.format(team=code, keeper=''), shirt_path(code)),
                (SHIRT.format(team=code, keeper='_1'), shirt_path(code, True))]
    for e in boot['elements']:
        if e['element_type'] in (1, 2, 3, 4) and e.get('photo'):
            pid = str(e['photo']).split('.')[0]
            out.append((PHOTO.format(id=pid), photo_path(e['photo'])))
    return out


def fetch_all(pause=PAUSE, boot=None):
    """Download every image not yet cached. Returns (fetched, missing)."""
    DIR.mkdir(parents=True, exist_ok=True)
    todo = [(u, p) for u, p in wanted(boot) if not p.exists()]
    # a new download invalidates its old thumbnails
    for _, p in todo:
        for old in DIR.glob(f'{p.stem}_w*.png'):
            old.unlink()
    fetched = missing = 0
    for i, (url, path) in enumerate(todo, 1):
        try:
            _download(url, path)
            fetched += 1
        except OSError:
            missing += 1           # no photo yet for new signings: placeholder
        time.sleep(pause)
        if i % 100 == 0:
            print(f'  images {i}/{len(todo)}', flush=True)
    return fetched, missing


def _thumbnail(path, width):
    """`width`-px copy of a cached image, itself cached on disk after the first call."""
    small = path.with_name(f'{path.stem}_w{width}.png')
    if not small.exists():
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((width, width * 2))
        img.save(small, format='PNG', optimize=True)
    return small.read_bytes()


def fetch(url, path):
    """Download one image into the cache now; False if it cannot be had."""
    if os.environ.get('FPL_OFFLINE') == '1' or not url:
        return False
    with _lock:
        if url in _state['failed'] or time.time() < _state['paused_until']:
            return False
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=RUNTIME_TIMEOUT) as r:
            data = r.read()
        tmp = path.with_name(f'{path.name}.{threading.get_ident()}.part')
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception:
        with _lock:
            _state['failed'].add(url)
            _state['streak'] += 1
            if _state['streak'] >= FAIL_LIMIT:
                _state['paused_until'] = time.time() + FAIL_COOLDOWN
                _state['streak'] = 0
        return False
    with _lock:
        _state['streak'] = 0
    return True


def placeholder(kind):
    svg = base64.b64encode(PLACEHOLDER[kind].encode()).decode()
    return f'data:image/svg+xml;base64,{svg}'


def uri(path, kind, width=None, url=None):
    """A data URI for an image, fetched on first use and shrunk to `width` px;
    the placeholder if it cannot be fetched. Only real images are memoised,
    so a placeholder is retried once the network is back."""
    memo = (path, width)
    if memo in _uris:
        return _uris[memo]
    if not path.exists() and not fetch(url, path):
        return placeholder(kind)
    try:
        data = _thumbnail(path, width) if width else path.read_bytes()
    except Exception:                     # a truncated or corrupt download
        return placeholder(kind)
    _uris[memo] = f'data:image/png;base64,{base64.b64encode(data).decode()}'
    return _uris[memo]


def photo(p, width=None):
    """Photo for a core.players() record."""
    pid = str(p.get('photo', '')).split('.')[0]
    return uri(photo_path(p.get('photo', '')), 'photo', width,
               PHOTO.format(id=pid) if pid else None)


def badge(p, width=None):
    code = p.get('team_code')
    return uri(badge_path(code), 'badge', width,
               BADGE.format(team=code) if code is not None else None)


def shirt(p, width=None):
    code, keeper = p.get('team_code'), p.get('element_type') == 1
    return uri(shirt_path(code, keeper), 'shirt', width,
               SHIRT.format(team=code, keeper='_1' if keeper else '')
               if code is not None else None)
