"""The tabs about the whole pool: Players, News, Build."""
import pandas as pd
import streamlit as st

import core
import images
from ui import ai, pitch, player, style as s

SORTS = {'Projection': 'Next GW', 'Form': 'Form', 'Price': 'Price',
         'Ownership': 'Owned %'}
NEWS_LIMIT = 60


def table(pool, proj):
    """One row per player: photo, badge, live FPL facts, the projection."""
    fpl = core.players()
    blend = {} if proj is None else \
        proj.drop_duplicates('code').set_index('code')['blended_ppg'].to_dict()
    mult = core.avail_map(pool['code'])
    rows = []
    for r in pool.drop_duplicates('code').itertuples():
        p = fpl.get(r.code)
        if p is None:
            continue
        ppg = blend.get(r.code, r.pred / 38)
        m = mult.get(r.code, [1.0] * 5)
        label = s.STATUS.get(p['status'], s.STATUS['a'])[0]
        rows.append({'code': r.code, 'Photo': images.photo(p, 30),
                     'Club': images.badge(p, 24), 'Player': p['name'],
                     'Team': p['club'], 'Pos': s.POS.get(p['element_type'], ''),
                     'Price': p['price'] / 10, 'Status': label,
                     'Next GW': ppg * m[0], 'Next 5 GWs': ppg * sum(m),
                     'Form': p['form'], 'Owned %': p['selected'],
                     'Points': p['total_points']})
    return pd.DataFrame(rows)


def players(pool, proj, offline):
    df = table(pool, proj)
    s.section('Player pool')
    c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 1.2])
    query = c1.text_input('Search', placeholder='Player name')
    pos = c2.multiselect('Position', ['GK', 'DEF', 'MID', 'FWD'])
    status = c3.multiselect('Status', sorted(df['Status'].unique()))
    sort = c4.selectbox('Sort by', list(SORTS))
    c1, c2 = st.columns([2, 3])
    clubs = c1.multiselect('Club', sorted(df['Team'].unique()))
    lo, hi = float(df['Price'].min()), float(df['Price'].max())
    price = c2.slider('Price', lo, hi, (lo, hi), step=0.5, format='£%.1fm')

    view = df[df['Price'].between(*price)]
    if query:
        view = view[view['Player'].str.contains(query, case=False, regex=False)]
    for col, chosen in (('Pos', pos), ('Status', status), ('Team', clubs)):
        if chosen:
            view = view[view[col].isin(chosen)]
    view = view.sort_values(SORTS[sort], ascending=False).reset_index(drop=True)

    st.caption(f'{len(view)} players. Select a row for details.')
    num = st.column_config.NumberColumn
    event = st.dataframe(
        view, key='players_table', hide_index=True, height=560, row_height=44,
        on_select='rerun', selection_mode='single-row', width='stretch',
        column_order=[c for c in view.columns if c != 'code'],
        column_config={'Photo': st.column_config.ImageColumn('', width='small'),
                       'Club': st.column_config.ImageColumn('', width='small'),
                       'Price': num(format='£%.1fm'),
                       'Next GW': num(format='%.1f pts'),
                       'Next 5 GWs': num(format='%.1f pts'),
                       'Form': num(format='%.1f'),
                       'Owned %': num(format='%.1f%%')})
    rows = event.selection.rows if event else []
    code = int(view.iloc[rows[0]]['code']) if rows else None
    # open the panel once per new selection, not on every rerun
    if code is not None and st.session_state.get('players_open') != code:
        player.show(code, pool, proj, offline)
    st.session_state['players_open'] = code


def _headline(h):
    title = s.esc(h['title'])
    if h.get('link'):
        title = f'<a href="{s.esc(h["link"])}" target="_blank">{title}</a>'
    return (f"<li>{title} <span class='sm-muted'>· {s.esc(h['source'])} · "
            f"{s.esc(h['date'][:16])}</span></li>")


def _news_html(code, tag, res):
    """One player's news card; `res` None means a rating is on its way."""
    p = core.players().get(code, {})
    if res is None:
        heads, verdict, rating = core.headlines(code), None, ai.spinner('Rating the news…')
    else:
        heads, verdict = res.get('headlines') or [], res.get('verdict')
        rating = ai.label(res) if 'label' in res else ''
    items = ''.join(_headline(h) for h in heads[:5])
    summary = verdict.get('summary') if verdict else None
    body = ''.join((
        f"<div class='sm-body'>FPL: {s.esc(p['news'])}</div>" if p.get('news') else '',
        f"<div class='sm-muted' style='margin-top:8px'>{s.esc(summary)}</div>"
        if summary and summary != 'No recent news' else '',
        f"<ul style='margin:10px 0 0 18px;padding:0'>{items}</ul>" if items else '',
        '' if (p.get('news') or heads) else "<div class='sm-muted'>No news.</div>"))
    return (f"<div class='sm-card'><div class='sm-row' style='align-items:flex-start'>"
            f"<img class='avatar' src='{images.photo(p, 112)}'><div style='flex:1'>"
            f"<div class='sm-spread'><div class='sm-row' style='gap:8px'>"
            f"<img class='crest' src='{images.badge(p, 44)}'>"
            f"<span class='sm-card-title'>{s.esc(p.get('name', code))}</span>"
            f"<span class='sm-muted'>{s.esc(p.get('club', ''))}{tag}</span></div>"
            f"<div class='sm-row' style='gap:6px'>"
            f"{s.status_badge(p.get('status', 'a'), p.get('chance'))}"
            f"{s.sentiment_badge(verdict)}{rating}</div></div>{body}</div></div></div>")


def news(pool, t, offline):
    s.section('News')
    choice = st.radio('Show', ['My squad', 'All', 'Injuries only'],
                      horizontal=True, label_visibility='collapsed')
    fpl = core.players()
    squad = [c for c in (t['codes'] if t else []) if c in fpl]
    rated = []
    if t and t['proj'] is not None:
        from ui import team
        free, hits, _ = team.plan_inputs(t)
        rated = [c for c in core.news_targets(t, core.transfer_plans(
            t['entry'], offline, free, hits)[0]) if c in fpl]
    targets = [c for c in rated if c not in squad]
    others = [c for c in pool['code'].drop_duplicates() if c in fpl
              and c not in rated and (fpl[c]['news'] or core.headlines(c))]
    severity = {'i': 0, 'u': 0, 's': 1, 'd': 2, 'n': 3, 'a': 4}
    others.sort(key=lambda c: (severity.get(fpl[c]['status'], 4), -fpl[c]['selected']))

    codes = squad + targets if choice == 'My squad' else squad + targets + others
    if choice == 'Injuries only':
        codes = [c for c in codes if fpl[c]['status'] != 'a']
    if not codes:
        s.card("<div class='sm-muted'>"
               + ('Load a team to see its news.' if choice == 'My squad'
                  else 'No news.') + '</div>')
        return
    if len(codes) > NEWS_LIMIT:
        st.caption(f'Showing the first {NEWS_LIMIT} of {len(codes)}: your squad and '
                   f'transfer targets first, then by severity and ownership.')
    st.caption('Ratings cover your squad and transfer targets. They are advisory '
               'and never change a suggestion.')
    slots, gw = ai.Slots(), (t['gw'] if t else None)
    for c in codes[:NEWS_LIMIT]:
        tag = ' · your squad' if c in squad else ' · transfer target' if c in targets else ''
        if c in rated:
            key = f'{gw}|news|{c}'
            slots.add([key], lambda r, c=c, tag=tag, key=key: _news_html(c, tag, r.get(key)))
        else:
            st.markdown(_news_html(c, tag, {'headlines': core.headlines(c),
                                            'verdict': core.verdict(c)}),
                        unsafe_allow_html=True)
    if rated:
        slots.run(core.news_jobs([c for c in codes[:NEWS_LIMIT] if c in rated], gw), gw)
