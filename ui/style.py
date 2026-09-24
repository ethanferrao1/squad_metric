"""Squadmetric's look: tokens, the one CSS block, and formatting helpers."""
import html

import streamlit as st

BRAND, TAGLINE = 'Squadmetric', 'Evidence-led squad and transfer analytics'

PAGE, SURFACE, RAISED = '#f5f7fa', '#ffffff', '#f1f5f9'
BORDER = 'rgba(15,23,42,0.10)'
TEXT, MUTED, FAINT = '#0f172a', '#475569', '#94a3b8'
ACCENT, ACCENT_SOFT = '#0d9488', 'rgba(13,148,136,0.10)'
# green / amber / red carry status and nothing else
GOOD, WARN, BAD = '#16a34a', '#d97706', '#dc2626'

POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

STATUS = {'a': ('Fit', GOOD), 'd': ('Doubtful', WARN), 'i': ('Injured', BAD),
          's': ('Suspended', BAD), 'u': ('Unavailable', FAINT), 'n': ('Unavailable', FAINT)}

VERDICT = {'strong': ('Strong', ACCENT), 'hold': ('Hold', MUTED),
           'injury cover': ('Injury cover', BAD),
           'substitute instead': ('Substitute instead', WARN),
           'below margin': ('Closest miss', FAINT)}

SENTIMENT = {'positive': GOOD, 'neutral': FAINT, 'negative': BAD}

# fixture difficulty on one teal ramp, light (easy) to dark (hard)
FDR = {1: ('#ccfbf1', '#134e4a'), 2: ('#5eead4', '#134e4a'),
       3: ('#e2e8f0', TEXT), 4: ('#475569', '#ffffff'), 5: ('#1e293b', '#ffffff')}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background: {PAGE}; color: {TEXT}; font-size: 16px; }}
/* the menu and deploy button only: the sidebar's expand control must stay */
#MainMenu, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; pointer-events: none; }}
header[data-testid="stHeader"] button {{ pointer-events: auto; }}
.block-container {{ max-width: 1200px; padding: 2.2rem 1.5rem 4rem; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; font-size: 14px; }}

.sm-header {{ display: flex; justify-content: space-between; align-items: flex-end;
  flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
.sm-brand {{ font-size: 30px; font-weight: 700; color: {TEXT}; line-height: 1.1; }}
.sm-tagline {{ font-size: 16px; color: {MUTED}; margin-top: 4px; }}
.sm-section {{ font-size: 20px; font-weight: 600; margin: 8px 0 12px; color: {TEXT}; }}
.sm-muted {{ color: {MUTED}; font-size: 14px; }}

[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
[data-baseweb="tab"] {{ font-weight: 600; padding: 10px 14px; border-radius: 8px 8px 0 0;
  transition: color .15s ease, background .15s ease; }}
[data-baseweb="tab"]:hover {{ background: {ACCENT_SOFT}; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT}; }}
[data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top: 24px; }}

.st-key-section {{ margin-bottom: 24px; }}
.st-key-section [role="radiogroup"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
.st-key-section label {{ padding: 10px 14px; margin: 0; border-radius: 8px 8px 0 0;
  font-weight: 600; transition: color .15s ease, background .15s ease; }}
.st-key-section label:hover {{ background: {ACCENT_SOFT}; }}
.st-key-section label:has(input:checked) {{ color: {ACCENT}; box-shadow: inset 0 -2px 0 {ACCENT}; }}
.st-key-section label > div:first-child {{ display: none; }}
.sm-spin {{ display: inline-block; width: 12px; height: 12px; margin-right: 8px;
  border: 2px solid rgba(148,163,184,.3); border-top-color: {ACCENT}; border-radius: 50%;
  vertical-align: -2px; animation: smspin .8s linear infinite; }}
@keyframes smspin {{ to {{ transform: rotate(360deg); }} }}
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {BORDER};
  border-radius: 14px; padding: 16px 20px; transition: all .15s ease; }}
[data-testid="stMetric"]:hover {{ border-color: {ACCENT}; transform: translateY(-2px); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 14px; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-size: 26px; font-weight: 600; color: {TEXT}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 14px; overflow: hidden; }}
[data-testid="stExpander"] details {{ border: 1px solid {BORDER}; border-radius: 14px; background: {SURFACE}; }}

.sm-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 20px 22px; margin: 0 0 24px; transition: border-color .15s ease,
  transform .15s ease, box-shadow .15s ease; }}
.sm-card:hover {{ border-color: rgba(20,184,166,.55); transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(15,23,42,.08); }}
.sm-card.best {{ border-color: {ACCENT}; box-shadow: inset 0 0 0 1px {ACCENT}; }}
.sm-card-title {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
.sm-row {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.sm-spread {{ display: flex; justify-content: space-between; align-items: center;
  gap: 12px; flex-wrap: wrap; }}
.sm-body {{ color: {TEXT}; margin-top: 12px; line-height: 1.55; }}
.sm-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 12px; margin: 16px 0 8px; }}
.sm-stat {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 12px; }}
.sm-stat .v {{ font-size: 19px; font-weight: 600; color: {TEXT}; }}
.sm-stat .l {{ font-size: 14px; color: {MUTED}; }}

.badge {{ display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px;
  border-radius: 999px; font-size: 13.5px; font-weight: 600; white-space: nowrap;
  background: rgba(148,163,184,.10); border: 1px solid {BORDER}; color: {TEXT}; }}
.avatar {{ width: 56px; height: 71px; object-fit: cover; border-radius: 10px;
  background: {RAISED}; border: 1px solid {BORDER}; }}
.avatar-lg {{ width: 110px; height: 140px; object-fit: cover; border-radius: 14px;
  background: {RAISED}; border: 1px solid {BORDER}; }}
.crest {{ width: 22px; height: 22px; vertical-align: middle; }}
.crest-lg {{ width: 44px; height: 44px; }}
.arrow {{ color: {ACCENT}; font-size: 22px; font-weight: 700; padding: 0 4px; }}
.fdr {{ display: inline-block; min-width: 86px; text-align: center; padding: 6px 8px;
  border-radius: 8px; font-weight: 600; font-size: 13.5px; margin: 0 6px 6px 0;
  border: 1px solid {BORDER}; }}

.pitch {{ border-radius: 16px 16px 0 0; padding: 14px 6px 6px;
  background: repeating-linear-gradient(180deg, #1f4d3a 0 46px, #1b4533 46px 92px);
  border: 1px solid {BORDER}; border-bottom: 0;
  box-shadow: inset 0 0 0 2px rgba(255,255,255,.05); }}
.bench {{ border-radius: 0 0 16px 16px; padding: 10px 6px 12px; background: #e2e8f0;
  border: 1px solid {BORDER}; margin-bottom: 24px; }}
.bench-label {{ color: {MUTED}; font-size: 13.5px; font-weight: 600; text-align: center;
  letter-spacing: .08em; margin-bottom: 4px; }}
.prow {{ display: flex; justify-content: space-evenly; align-items: flex-start; padding: 8px 0; }}
.pl {{ position: relative; width: 96px; text-align: center; transition: transform .15s ease; }}
.pl:hover {{ transform: translateY(-3px); }}
.pl img {{ width: 54px; height: auto; filter: drop-shadow(0 5px 6px rgba(0,0,0,.35)); }}
.pl .mk {{ position: absolute; top: -2px; right: 14px; width: 20px; height: 20px;
  border-radius: 50%; background: {TEXT}; color: #ffffff; font-size: 11px; font-weight: 800;
  line-height: 20px; box-shadow: 0 2px 6px rgba(0,0,0,.4); }}
.pl .mk.vc {{ background: {RAISED}; color: {TEXT}; border: 1px solid {MUTED}; line-height: 18px; }}
.pl .nm {{ background: rgba(11,18,32,.88); color: #fff; font-size: 13px; font-weight: 600;
  border-radius: 6px 6px 0 0; padding: 3px 4px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; margin-top: 2px; }}
.pl .pj {{ background: {ACCENT}; color: #ffffff; font-size: 13px; font-weight: 700;
  border-radius: 0 0 6px 6px; padding: 2px 4px; }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 4px; vertical-align: middle; }}
.sm-tiles {{ display: flex; gap: 8px; }}
.sm-tile {{ flex: 1 1 0; min-width: 0; background: {SURFACE}; border: 1px solid {BORDER};
  border-radius: 10px; padding: 6px 10px; }}
.sm-tile .v {{ font-size: 16px; font-weight: 600; color: {TEXT}; white-space: nowrap; }}
.sm-tile .l {{ font-size: 12px; color: {MUTED}; }}

/* narrow screens: the top bar replaces the sidebar's controls; tabs scroll */
.st-key-mobilebar {{ display: none !important; }}
@media (max-width: 899px) {{
  .st-key-mobilebar {{ display: flex !important; margin-bottom: 4px; }}
  .st-key-mobileteam {{ flex-wrap: nowrap !important; }}
  .st-key-mobileteam > div:first-child {{ flex: 1 1 0 !important; min-width: 0; width: auto !important; }}
  .st-key-mobileteam > div:last-child {{ flex: 0 0 auto !important; width: auto !important; }}
  .block-container {{ padding-top: 3.4rem; }}
  .st-key-section {{ margin-bottom: 16px; }}
  .st-key-section [role="radiogroup"] {{ flex-wrap: nowrap; overflow-x: auto;
    scrollbar-width: none; -webkit-overflow-scrolling: touch; }}
  .st-key-section [role="radiogroup"]::-webkit-scrollbar {{ display: none; }}
  .st-key-section label {{ flex: 0 0 auto; white-space: nowrap; padding: 8px 12px; }}
}}
@media (max-width: 640px) {{
  .block-container {{ padding: 3.4rem .6rem 3rem; }}
  .pl {{ width: 60px; }} .pl img {{ width: 36px; }}
  .pl .nm, .pl .pj {{ font-size: 11px; }} .pl .mk {{ right: 2px; }}
  .sm-card {{ padding: 16px; }} .sm-brand {{ font-size: 24px; }}
}}
</style>
"""


def inject():
    """The one CSS block, once per run."""
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def money(tenths):
    """£6.5m from FPL's tenths of a million."""
    return f'£{tenths / 10:.1f}m'


def money_signed(tenths):
    sign = '+' if tenths > 0 else '−' if tenths < 0 else ''
    return sign + money(abs(tenths))


def num(x, signed=False):
    if x is None or x != x:
        return '–'
    return f'{x:+.1f}' if signed else f'{x:.1f}'


def pts(x, signed=False):
    """5.2 pts."""
    return '–' if x is None or x != x else f'{num(x, signed)} pts'


def pct(x):
    return f'{x:.1f}%'


def badge(text, colour=None):
    style = (f' style="color:{colour};border-color:{colour}55;'
             f'background:{colour}1a"' if colour else '')
    return f'<span class="badge"{style}>{text}</span>'


def status_badge(status, chance=None):
    label, colour = STATUS.get(status, STATUS['a'])
    if status == 'd' and chance is not None:
        label = f'{label} {chance}%'
    return badge(label, colour)


def status_dot(status):
    return (f'<span class="dot" style="background:'
            f'{STATUS.get(status, STATUS["a"])[1]}"></span>')


def verdict_badge(verdict):
    return badge(*VERDICT.get(verdict, VERDICT['hold']))


def sentiment_badge(v):
    if not v or v.get('summary') == 'No recent news':
        return ''
    flags = f" · {', '.join(v['flags'])}" if v['flags'] else ''
    return badge(f"News: {v['sentiment']}{flags}",
                 SENTIMENT.get(v['sentiment'], FAINT))


def fdr_chip(gw, opp, home, difficulty):
    bg, fg = FDR.get(difficulty, FDR[3])
    return (f'<span class="fdr" style="background:{bg};color:{fg}">'
            f'GW{gw} · {esc(opp)} ({"H" if home else "A"})</span>')


def header(gw, refreshed, offline):
    meta = [badge(f'GW{gw} data'), badge(f'Refreshed {esc(refreshed)}')]
    if offline:
        meta.append(badge('Offline mode', WARN))
    st.markdown(f"<div class='sm-header'><div><div class='sm-brand'>{BRAND}</div>"
                f"<div class='sm-tagline'>{TAGLINE}</div></div>"
                f"<div class='sm-row'>{''.join(meta)}</div></div>",
                unsafe_allow_html=True)


def section(title):
    st.markdown(f"<div class='sm-section'>{title}</div>", unsafe_allow_html=True)


def card(body_html):
    st.markdown(f"<div class='sm-card'>{body_html}</div>", unsafe_allow_html=True)


def tiles(items):
    """Small labelled figures in one row: the narrow-screen top bar."""
    cells = ''.join(f"<div class='sm-tile'><div class='l'>{esc(l)}</div>"
                    f"<div class='v'>{v}</div></div>" for l, v in items)
    return f"<div class='sm-tiles'>{cells}</div>"


def stats(items):
    """A responsive grid of small labelled figures."""
    cells = ''.join(f"<div class='sm-stat'><div class='v'>{v}</div>"
                    f"<div class='l'>{esc(l)}</div></div>" for l, v in items)
    return f"<div class='sm-stats'>{cells}</div>"


def plot_layout(fig, title, height=240):
    """Transparent, gridless, hover-only."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT), x=0, xanchor='left'),
        template='plotly_white', height=height, showlegend=False,
        margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, system-ui, sans-serif', size=14, color=MUTED),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=ACCENT, font=dict(size=14, color=TEXT)))
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
    return fig


def chart(container, fig):
    container.plotly_chart(fig, width='stretch',
                           config={'displayModeBar': False})
