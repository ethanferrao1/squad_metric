"""The app's data and suggestion logic, with no page layout.

Moved out of app.py unchanged so the UI in ui/ can be redesigned freely. Online
by default: team ids and player histories are fetched on demand and cached;
if the network fails it falls back to the cache. FPL_OFFLINE=1 never touches
the network.
"""
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import build_live
import fpl_api as api
import optimise as op
import rater as rt

# the demo switch: cache only, no network at all
OFFLINE = os.environ.get('FPL_OFFLINE') == '1'


@st.cache_data(show_spinner=False)
def load_pool():
    return build_live.load()


class NoPicks(Exception):
    """The entry exists but has no picks for this gameweek yet."""


@st.cache_data(show_spinner=False)
def load_team(entry_id, offline=False):
    """(codes, bank, gameweek, from_cache) for an entry.

    Live for this gameweek when possible, cached for next time. Offline, or
    if the fetch fails, the latest gameweek already cached for that entry. An
    unknown ID raises the API's 404; a team with no picks yet raises NoPicks.
    """
    import urllib.error
    gw = api.current_gw()
    if not offline:
        try:
            codes, bank = api.load_team(int(entry_id), gw)
            return codes, bank, gw, False
        except urllib.error.HTTPError as e:
            if e.code == 404 and api.cached_event(int(entry_id)) is None:
                api._get(f'entry/{int(entry_id)}')      # raises 404 for an unknown ID
                raise NoPicks(f'Team {entry_id} has no picks for GW{gw} yet: a new '
                              f'team appears after its first deadline.')
        except OSError:
            pass
    cached = api.cached_event(int(entry_id))
    if cached is None:
        raise FileNotFoundError(f'team {entry_id} is not cached and the '
                                f'network is unavailable')
    codes, bank = api.load_team(int(entry_id), cached)
    return codes, bank, cached, cached != gw or not offline


@st.cache_data(show_spinner=False)
def blended(_pool, codes, offline=False):
    """form_lab's blended projection for the squad and its swap candidates.

    The blend won the three-season simulation, so the Improve tab and the
    captain pick run on it. Build stays on the season model. Any history not
    yet cached is fetched unless `offline`. Returns (proj, gw, fell_back).
    """
    from form_lab import data as fdata, form as fform

    gw = api.current_gw()
    fdata.STATUS['network_failed'] = False
    # every buyable player, not a top-N shortlist: cheap replacements matter,
    # and refresh.py keeps all of their histories cached
    buyable = rt.available(rt._modelled(_pool))['code']
    wanted = sorted(set(codes) | set(buyable))
    hist = fdata.live_history(wanted, offline=offline)
    fell_back = fdata.STATUS['network_failed']
    if hist.empty:
        return None, gw, fell_back

    by_code = _pool.drop_duplicates('code').set_index('code')
    snap = fform.snapshot(hist, gw + 1, by_code['pred'])
    snap['price'] = snap['code'].map(by_code['price'])
    snap['has_fixture'] = snap['code'].map(
        fdata.has_fixture(snap['code'], gw + 1))
    proj = fform.project(snap)

    # a squad player with no history to read keeps his season-model
    # projection; dropping him would value a 14-man squad and inflate gains
    missing = [c for c in codes
               if c not in set(proj['code']) and c in by_code.index]
    if missing:
        extra = pd.DataFrame({'code': missing})
        extra['blended_ppg'] = (extra['code'].map(by_code['pred'])
                                / fform.SEASON_GWS)
        extra['has_fixture'] = extra['code'].map(
            fdata.has_fixture(missing, gw + 1))
        proj = pd.concat([proj, extra], ignore_index=True)

    proj['web_name'] = proj['code'].map(by_code['web_name'])
    proj['regular'] = proj['code'].isin(
        fdata.live_regulars(list(proj['code']), gw))
    return proj, gw, fell_back


@st.cache_data(show_spinner=False)
def availability_info():
    """code -> FPL status, chance, news and per-gameweek availability."""
    import availability as av
    return av.lookup(api.bootstrap()['elements'])


def avail_map(codes):
    """code -> per-gameweek availability multipliers, next gameweek first."""
    info = availability_info()
    return {c: info[c]['mult'] for c in codes if c in info}


@st.cache_data(show_spinner=False)
def optimal_squad(_pool, offline=False):
    """MILP squad over the modelled, available players, at the standard
    100.0m. Anyone FPL rates below 50% to play is not buyable, and the bench
    is only regular starters (60+ min in half their team's last 5 games)."""
    from form_lab import backtest as fbt, data as fdata

    cands = rt.available(rt._modelled(_pool))
    regulars = fdata.live_regulars(list(cands['code']), api.current_gw(),
                                   offline=offline)
    try:
        squad = fbt.pick_squad(cands, 'pred', regulars)
    except RuntimeError:
        # no cached live history to judge starters by: the plain MILP
        squad = op.pick_squad(cands, 'pred')
    xi, captain = op.best_xi(squad, 'pred')
    return squad, xi, captain


def data_stamp():
    """(gameweek, 'refreshed ...' text) from refresh.py's stamp, else the
    bootstrap cache's file time."""
    import refresh
    s = refresh.stamp()
    if s:
        when = datetime.fromisoformat(s['refreshed']).astimezone()
        return s['gw'], when.strftime('%d %b %H:%M')
    f = api.CACHE_DIR / 'bootstrap-static.json'
    when = datetime.fromtimestamp(f.stat().st_mtime) if f.exists() else None
    return api.current_gw(), when.strftime('%d %b %H:%M') if when else 'never'


def money(tenths):
    return f'£{tenths / 10:.1f}m'


def template_reason(row):
    """One line per suggested transfer: the fallback when no live text is available."""
    from form_lab import form as fform
    cost = row['cost']
    if cost > 0:
        money_part = f'for {money(cost)} more'
    elif cost < 0:
        money_part = f'and frees {money(-cost)}'
    else:
        money_part = 'at the same price'
    return (f"{row['in']} is projected {row['gain']:+.0f} pts over {row['out']} "
            f"across {fform.SWAP_HORIZON} gameweeks {money_part}.")


def explain(row, pool=None, use_llm=False, model=None):
    """(text, source, detail) -- why this transfer is worth making.

    `source` is 'llm', 'cache' or 'template'; `detail` is the model that
    wrote it, or the reason the template was used instead. The LLM only
    rewrites prose; every number is checked back against the facts, and any
    failure lands on the template. `model` overrides the default free model
    and turns off the free fallbacks.
    """
    template = template_reason(row)
    if not use_llm or pool is None:
        return template, 'template', 'LLM explanations off'

    import llm_explain
    over = {'model': model, 'models': None} if model else {}
    return llm_explain.explain(row, pool, template, **over)


def pick_captain(xi, proj, out):
    """form_lab's captain on next gameweek's availability; never anyone out."""
    fit = xi[~xi['code'].isin(out)]
    if proj is not None:
        from form_lab import form as fform
        nxt = {c: m[0] for c, m in avail_map(proj['code']).items()}
        adj = proj.assign(blended_ppg=proj['blended_ppg']
                          * proj['code'].map(nxt).fillna(1.0))
        cap = fform.captain(fit, adj, unavailable=out)
        if cap is not None:
            return cap
    return fit.nlargest(1, 'pred')['code'].iloc[0] if len(fit) else None


def rate_now(pool, codes, bank, proj=None):
    """rate_squad on availability.

    Season projections lose the gameweeks a player will miss; the XI is only
    players who can play next gameweek; the captain is never unavailable.
    """
    import availability as av
    info = availability_info()
    adj = pool.copy()
    adj['pred'] = adj['pred'] * adj['code'].map(
        {c: av.season_factor(i) for c, i in info.items()}).fillna(1.0)
    r = rt.rate_squad(codes, adj, bank=bank)

    sq = r['players'].copy()
    out = {c for c in sq['code'] if c in info and av.out_next(info[c])}
    fit_xi, _ = op.best_xi(sq, 'pred')
    xi, _ = op.best_xi(sq.assign(pick=sq['pred'].where(~sq['code'].isin(out),
                                                       -1e9)), 'pick')
    cap = pick_captain(xi, proj, out)
    playing = xi[~xi['code'].isin(out)]
    score = playing['pred'].sum() + sq.loc[sq['code'] == cap, 'pred'].sum()

    sq['in_xi'] = sq['code'].isin(xi['code'])
    sq['captain'] = sq['code'] == cap
    r.update(players=sq, captain=cap, score=score,
             pct=100 * score / r['optimum'], unavailable=out,
             injured_starters=out & set(fit_xi['code']),
             # the bench players who start in their place: the free fix
             sub_ins=set(playing['code']) - set(fit_xi['code']))
    return r


def swap_rows(pool, r, proj):
    """(swaps, below_margin) on the availability-adjusted blend over 5 GWs.

    Any starter who cannot play next gameweek comes first, with his best
    replacement. The gain is measured against starting the bench player in
    his place, so the row is 'injury cover' only if the transfer beats that
    free substitution by SWITCH_MARGIN, else 'substitute instead'.
    `below_margin` means no other swap clears the margin.
    """
    from form_lab import form as fform

    codes = list(r['players']['code'])
    kw = dict(bank=r['bank'], margin=fform.SWITCH_MARGIN,
              eligible=set(proj.loc[proj['regular'], 'code']),
              avail=avail_map(set(codes) | set(proj['code'])))

    cover = pd.DataFrame()
    if r.get('injured_starters'):
        cover = fform.suggest_swaps(codes, proj, pool, fform.SWAP_HORIZON,
                                    include_below=True, k=99, per_slot=5,
                                    only_out=r['injured_starters'], **kw)
        if len(cover):
            cover = cover.sort_values('gain_pts', ascending=False) \
                .drop_duplicates('out')
            cover['verdict'] = [fform.cover_verdict(g) for g in cover['gain_pts']]

    swaps = fform.suggest_swaps(codes, proj, pool, fform.SWAP_HORIZON, **kw)
    below = not len(swaps)
    if below:
        swaps = fform.suggest_swaps(codes, proj, pool, fform.SWAP_HORIZON,
                                    include_below=True, **kw)
    if len(cover) and len(swaps):
        swaps = swaps[~swaps['out'].isin(cover['out'])
                      & ~swaps['in'].isin(cover['in'])]
    return pd.concat([cover, swaps], ignore_index=True), below


def news_for(name, proj):
    """Cached news sentiment for one player, or None when nothing is stored."""
    hit = proj.loc[proj['web_name'] == name, 'code']
    return verdict(int(hit.iloc[0])) if len(hit) else None


CHIP_NOTES = {
    'freehit': 'Free hit active in GW{gw}: this is the one-week squad, and your usual '
               'team returns next week. Plan transfers from that team, not this one.',
    'bboost': 'Bench boost active in GW{gw}: every player scores this week.',
    '3xc': 'Triple captain active in GW{gw}.',
    'wildcard': 'Wildcard active in GW{gw}: transfers this week are free.',
}


@st.cache_data(show_spinner='Loading team…')
def team_analysis(entry, offline=False):
    """Everything the team tabs show for one entry, computed once."""
    import lineup

    pool = load_pool()
    codes, bank, team_gw, fell_back = load_team(entry, offline)
    proj, gw, hist_fell_back = blended(pool, tuple(codes), offline)
    r = rate_now(pool, codes, bank, proj)
    out = dict(entry=entry, codes=codes, bank=bank, team_gw=team_gw, gw=gw,
               fell_back=fell_back or hist_fell_back, proj=proj, r=r,
               swaps=pd.DataFrame(), below=True, lineup=None)
    if proj is None:
        return out

    out['swaps'], out['below'] = swap_rows(pool, r, proj)
    pk = picks(entry, team_gw)
    out['chip_note'] = CHIP_NOTES.get(api.picks(int(entry), team_gw).get('active_chip'), '') \
        .format(gw=team_gw)
    base, play = next_gw_play(proj, codes)
    pos = pool.drop_duplicates('code').set_index('code')['element_type']
    current = {'xi': list(pk.loc[pk['slot'] <= 11, 'code']),
               'bench': list(pk.loc[pk['slot'] > 11, 'code']),
               'captain': pk.loc[pk['captain'], 'code'].iloc[0],
               'vice': pk.loc[pk['vice'], 'code'].iloc[0]}
    out.update(picks=pk, base=base, play=play, current=current,
               lineup=lineup.compare(current, base, play,
                                     {c: int(pos[c]) for c in codes}))
    return out


from form_lab import backtest as _bt, freehit as _fh   # noqa: E402

BUILD_VIEWS = api.CACHE_DIR / 'build_views.pkl'
FH_THRESHOLD, FH_MIN_BLANK = _bt.FREE_HIT['threshold'], _fh.MIN_BLANK


def free_transfers(entry_id, gw):
    """Free transfers for the next deadline, rebuilt from the entry's history; None if unknown."""
    try:
        h = api._get(f'entry/{entry_id}/history')
    except OSError:
        return None
    made = {r['event']: r['event_transfers'] for r in h.get('current', [])}
    if not made:
        return None
    chips = {c['event']: c['name'] for c in h.get('chips', [])}
    ft = 1
    for g in range(min(made) + 1, gw + 1):
        spent = 0 if chips.get(g) in ('wildcard', 'freehit') else made.get(g, 0)
        ft = min(_bt.MAX_FREE, max(0, ft - spent) + 1)
    return ft


def _market(pool, t):
    """What the planners need: 5-GW values, prices, positions, clubs, who is out."""
    import availability as av
    by = pool.drop_duplicates('code').set_index('code')
    blend = t['proj'].drop_duplicates('code').set_index('code')['blended_ppg']
    info = availability_info()
    buyable = set(rt.available(rt._modelled(pool))['code'])
    codes = (buyable | set(t['codes'])) & set(blend.index)
    mult = {c: m for c, m in avail_map(codes).items()}
    value = {c: float(blend[c]) * sum(mult.get(c, [1.0] * 5)) for c in codes}
    out = {c for c in codes if c in info and av.out_next(info[c])}
    kw = dict(value=value, prices=by['price'].astype(int).to_dict(),
              pos=by['element_type'].astype(int).to_dict(),
              clubs=by['team_code'].to_dict(),
              eligible=set(t['proj'].loc[t['proj']['regular'], 'code']),
              unavailable=out)
    nxt = {c: float(blend[c]) * mult.get(c, [1.0])[0] for c in codes}
    return kw, nxt


@st.cache_data(show_spinner='Planning transfers…')
def transfer_plans(entry, offline, free, hits):
    """{transfer count: plan} and the best count, from form_lab.plan's MILP.

    Owned players are valued at today's price: the public API does not give
    selling prices.
    """
    from form_lab import plan
    pool, t = load_pool(), team_analysis(entry, offline)
    m, _ = _market(pool, t)
    sells = {c: m['prices'].get(c, 0) for c in t['codes']}
    best, by = plan.transfer_plan(t['codes'], t['r']['bank'], sells, free, hits, **m)
    return {n: p for n, p in by.items() if p}, best['transfers']


@st.cache_data(show_spinner='Building a wildcard squad…')
def wildcard_now(entry, offline):
    """The squad a wildcard would build now (informational), and its 5-GW gain."""
    from form_lab import plan
    pool, t = load_pool(), team_analysis(entry, offline)
    m, _ = _market(pool, t)
    sells = {c: m['prices'].get(c, 0) for c in t['codes']}
    wc = plan.wildcard_plan(t['codes'], t['r']['bank'], sells, **m)
    squad = sorted((set(t['codes']) - set(wc['outs'])) | set(wc['ins']))
    base, _ = plan.transfer_plan(t['codes'], t['r']['bank'], sells, 0, 0,
                                 by_count=False, **m)
    return {'squad': squad, 'xi': sorted(wc['xi']), 'value': wc['value'],
            'gain': wc['value'] - base['value'], 'ins': wc['ins'], 'outs': wc['outs']}


@st.cache_data(show_spinner=False)
def free_hit_now(entry, offline):
    """The adopted free-hit rule for next gameweek: blank starters, then FH value >= 15."""
    from form_lab import backtest as fbt, data as fdata
    pool, t = load_pool(), team_analysis(entry, offline)
    if not t.get('current'):
        return None
    gw = t['gw'] + 1
    teams = fdata.teams_with_fixture(gw)
    fixtures = {}
    for f in fdata.fixtures():
        if f.get('event') == gw:
            for side in ('team_h', 'team_a'):
                fixtures[f[side]] = fixtures.get(f[side], 0) + 1
    fpl = players()
    starters = t['current']['xi']
    blank = [c for c in starters if fpl.get(c, {}).get('team') not in teams]
    out = {'gw': gw, 'blank': blank, 'play': False}
    if len(blank) < FH_MIN_BLANK:
        return out

    m, nxt = _market(pool, t)
    one = {c: v * fixtures.get(fpl.get(c, {}).get('team'), 0) for c, v in nxt.items()}
    cands = pd.DataFrame({'code': [c for c in one
                                   if c in t['codes'] or c not in m['unavailable']]})
    cands['element_type'] = cands['code'].map(m['pos'])
    cands['team_code'] = cands['code'].map(m['clubs'])
    cands['price'] = cands['code'].map(m['prices'])
    cands['pred'] = cands['code'].map(one)
    spend = t['r']['bank'] + sum(m['prices'].get(c, 0) for c in t['codes'])
    new = fbt.pick_squad(cands.dropna(), 'pred', m['eligible'], budget=int(spend))
    fh = [int(c) for c in new['code']]
    gain = fbt.xi_value(fh, one, m['pos']) - fbt.xi_value(t['codes'], one, m['pos'])
    out.update(squad=fh, xi=sorted(fbt.xi_members(fh, one, m['pos'])), gain=gain,
               values=one, play=gain >= FH_THRESHOLD)
    return out


def plan_pairs(outs, ins):
    """A plan's outs and ins matched by position, for display and explanation."""
    fpl = players()
    ins, pairs = list(ins), []
    for o in outs:
        i = next((c for c in ins if fpl.get(c, {}).get('element_type')
                  == fpl.get(o, {}).get('element_type')), ins[0])
        ins.remove(i)
        pairs.append((o, i))
    return pairs


def explain_jobs(t, pool, plans):
    """Live-AI explanation jobs for this team's single swaps, injury cover and plan moves."""
    import live_ai
    from form_lab import backtest as fbt
    gw, proj = t['gw'], t['proj']
    blend = proj.drop_duplicates('code').set_index('code')['blended_ppg']
    fpl, jobs = players(), []
    swaps = t['swaps']
    for _, r in (swaps.iterrows() if len(swaps) else []):
        if r['verdict'] == 'substitute instead':
            continue
        row = r.to_dict()
        jobs.append({'key': f"{gw}|swap|{row['out']}|{row['in']}", 'kind': 'explain',
                     'subject': f"{row['out']}->{row['in']}", 'pool': pool,
                     'facts': live_ai.explain_facts(row, pool, proj),
                     'template': template_reason(row)})
    for n, p in plans.items():
        net = p['gain'] - fbt.HIT_COST * p['hits']
        for o, i in plan_pairs(p['outs'], p['ins']):
            row = {'out': fpl[o]['name'], 'in': fpl[i]['name'],
                   'out_pred': float(blend.get(o, 0)), 'in_pred': float(blend.get(i, 0)),
                   'gain': net, 'cost': fpl[i]['price'] - fpl[o]['price']}
            jobs.append({'key': f"{gw}|plan{n}|{row['out']}|{row['in']}", 'kind': 'explain',
                         'subject': f"{row['out']}->{row['in']} (plan {n})", 'pool': pool,
                         'facts': live_ai.explain_facts(row, pool, proj,
                                                        {'transfers': n, 'net': net}),
                         'template': f"Part of a {n}-transfer plan worth {net:+.0f} "
                                     f"pts across {fbt.form.SWAP_HORIZON} gameweeks."})
    return jobs


def news_targets(t, plans):
    """The squad plus every incoming transfer target, in that order."""
    ins = [] if not len(t['swaps']) else [
        c for c in t['swaps']['in'].map(
            pd.Series({v['name']: k for k, v in players().items()})) if c == c]
    ins += [c for p in plans.values() for c in p['ins']]
    return list(dict.fromkeys([*t['codes'], *[int(c) for c in ins]]))


def news_jobs(codes, gw):
    """Live-AI news-rating jobs for these players."""
    from form_lab import sentiment as fs
    jobs = []
    for c in codes:
        try:
            full, _, web = fs._player(int(c))
        except KeyError:
            continue
        jobs.append({'key': f'{gw}|news|{c}', 'kind': 'news', 'code': int(c),
                     'subject': web, 'names': (web, full)})
    return jobs


def compute_build_views(pool, offline):
    """The live replay behind the Build tab; refresh.py stores it."""
    from form_lab import live
    return live.replay(pool, offline)


@st.cache_data(show_spinner='Replaying the season…')
def build_views(offline):
    """Stored views from refresh.py, else computed now."""
    if BUILD_VIEWS.exists():
        return pd.read_pickle(BUILD_VIEWS)
    return compute_build_views(load_pool(), offline)


# ---------------------------------------------------------------- read-only helpers for the UI

@st.cache_data(show_spinner=False)
def players():
    """code -> live FPL facts: club, position, status, price, season stats."""
    boot = api.bootstrap()
    clubs = {t['id']: t['short_name'] for t in boot['teams']}
    return {e['code']: {
        'id': e['id'], 'name': e['web_name'], 'club': clubs.get(e['team'], ''),
        'team': e['team'], 'team_code': e.get('team_code'),
        'photo': e.get('photo', ''), 'element_type': e['element_type'],
        'status': e['status'], 'chance': e['chance_of_playing_next_round'],
        'news': (e['news'] or '').strip(), 'price': e['now_cost'],
        'selected': float(e['selected_by_percent'] or 0),
        'form': float(e['form'] or 0), 'total_points': e['total_points'],
        'ppg': float(e['points_per_game'] or 0), 'minutes': e['minutes'],
        'goals': e['goals_scored'], 'assists': e['assists'],
        'clean_sheets': e['clean_sheets'], 'bonus': e['bonus'],
    } for e in boot['elements']}


def picks(entry_id, gw):
    """The entry's lineup for `gw`: code, slot 1-15, captain, vice."""
    lut = api.element_codes()
    rows = [{'code': lut[p['element']], 'slot': p['position'],
             'captain': p['is_captain'], 'vice': p['is_vice_captain']}
            for p in api.picks(entry_id, gw)['picks'] if p['element'] in lut]
    return pd.DataFrame(rows).sort_values('slot').reset_index(drop=True)


def next_gw_play(proj, codes):
    """(points if he plays, chance he plays) next gameweek, per code.

    The chance is FPL's availability for next gameweek, and zero for a blank.
    """
    by = proj.drop_duplicates('code').set_index('code')
    mult = avail_map(codes)
    base = {c: float(by['blended_ppg'].get(c, 0.0)) for c in codes}
    fixture = by['has_fixture'].fillna(False) if 'has_fixture' in by else {}
    play = {c: mult.get(c, [1.0])[0] * float(bool(fixture.get(c, True)))
            for c in codes}
    return base, play


def history(code, offline):
    """One player's gameweek history (points, minutes, price), fetched if needed."""
    from form_lab import data as fdata
    return fdata.live_history([code], offline=offline).sort_values('gw')


def projection(pool, proj, code, offline):
    """prior, form and blended points per gameweek for one player."""
    if proj is not None and code in set(proj['code']):
        row = proj.loc[proj['code'] == code].iloc[0]
        return {k: row.get(k) for k in ('prior_ppg', 'form_ppg', 'blended_ppg')}

    from form_lab import form as fform
    by = pool.drop_duplicates('code').set_index('code')
    prior = by['pred'].get(code)
    hist = history(code, offline)
    if hist.empty or prior is None:
        p = None if prior is None else prior / fform.SEASON_GWS
        return {'prior_ppg': p, 'form_ppg': None, 'blended_ppg': p}
    snap = fform.snapshot(hist, api.current_gw() + 1, {code: prior})
    row = fform.project(snap).iloc[0]
    return {k: row.get(k) for k in ('prior_ppg', 'form_ppg', 'blended_ppg')}


def fixtures_ahead(team_id, gw, n=5):
    """[(gw, opponent, home, FPL difficulty)] for his club's next `n` gameweeks."""
    from form_lab import data as fdata
    clubs = {t['id']: t['short_name'] for t in api.bootstrap()['teams']}
    out = []
    for f in fdata.fixtures():
        ev = f.get('event')
        if ev is None or not gw < ev <= gw + n:
            continue
        if f['team_h'] == team_id:
            out.append((ev, clubs.get(f['team_a'], '?'), True,
                        f['team_h_difficulty']))
        elif f['team_a'] == team_id:
            out.append((ev, clubs.get(f['team_h'], '?'), False,
                        f['team_a_difficulty']))
    return sorted(out)


def match_log(code, offline):
    """Per gameweek: points, minutes and opponents; NaN points for a blank or before he joined."""
    from form_lab import data as fdata
    p = players().get(code)
    gws = range(1, api.current_gw() + 1)
    empty = pd.DataFrame({'gw': list(gws), 'points': float('nan'), 'minutes': 0,
                          'opp': ''})
    if p is None or (offline and not fdata.is_cached(p['id'])):
        return empty
    try:
        hist = fdata.element_summary(p['id'])['history']
    except OSError:
        return empty
    clubs = {t['id']: t['short_name'] for t in api.bootstrap()['teams']}
    rows = pd.DataFrame([{'gw': h['round'], 'points': h['total_points'],
                          'minutes': h['minutes'],
                          'opp': f"{clubs.get(h['opponent_team'], '?')} "
                                 f"({'H' if h['was_home'] else 'A'})"} for h in hist])
    if rows.empty:
        return empty
    per = rows.groupby('gw').agg(points=('points', 'sum'), minutes=('minutes', 'sum'),
                                 opp=('opp', ' + '.join))
    return per.reindex(list(gws)).rename_axis('gw').reset_index() \
        .fillna({'minutes': 0, 'opp': ''})


def headlines(code):
    """Cached Google News items for a player ([] if none were fetched)."""
    from form_lab import sentiment as fsent
    f = fsent.NEWS_DIR / f'{code}.json'
    return json.loads(f.read_text(encoding='utf-8'))['headlines'] \
        if f.exists() else []


def verdict(code):
    """Cached LLM news verdict for a player, or None."""
    from form_lab import sentiment as fsent
    f = fsent.VERDICT_DIR / f'{code}.json'
    return json.loads(f.read_text(encoding='utf-8'))['verdict'] \
        if f.exists() else None
