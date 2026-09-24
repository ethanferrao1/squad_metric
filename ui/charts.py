"""Plotly figures, built from plain frames so they can be tested without a page."""
import plotly.graph_objects as go

from ui import style as s

ROLLING = 10


def _series(fig, log, projection, name, colour, legend):
    played = log.dropna(subset=['points'])
    fig.add_trace(go.Scatter(
        x=log['gw'], y=log['points'], mode='lines+markers', name=name,
        connectgaps=False, line=dict(color=colour, width=2), marker=dict(size=7),
        customdata=log[['opp', 'minutes']].to_numpy(), showlegend=legend,
        hovertemplate='GW%{x} · %{customdata[0]}<br>%{customdata[1]} min · '
                      '%{y} pts<extra>' + s.esc(name) + '</extra>'))
    if len(played):
        roll = played['points'].rolling(ROLLING, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=played['gw'], y=roll, mode='lines', name=f'{name} 10-game average',
            line=dict(color=colour, width=1.5, dash='dash'), showlegend=False,
            hovertemplate='GW%{x}: 10-game average %{y:.1f} pts<extra></extra>'))
    if projection is not None and projection == projection:
        fig.add_trace(go.Scatter(
            x=[log['gw'].min(), log['gw'].max() + 1], y=[projection] * 2,
            mode='lines', name=f'{name} projection', showlegend=False,
            line=dict(color=colour, width=1.5, dash='dot'),
            hovertemplate=f'Projection {projection:.1f} pts per game<extra></extra>'))


def points_chart(log, projection=None, name='', compare=None):
    """Points per gameweek (gaps for blanks), dashed 10-game average, dotted projection.

    `compare` is another (log, projection, name) drawn over it in grey.
    """
    fig = go.Figure()
    title = 'Points per gameweek'
    if log is None or log.empty or log['points'].isna().all():
        fig.add_annotation(text='No appearances yet', showarrow=False,
                           font=dict(color=s.MUTED, size=15))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        _series(fig, log, projection, name, s.ACCENT, compare is not None)
    if compare is not None:
        _series(fig, *compare, s.MUTED, True)
        title = f'{name} vs {compare[2]}'
    fig = s.plot_layout(fig, title, height=280)
    fig.update_layout(showlegend=compare is not None,
                      legend=dict(orientation='h', y=-0.2, x=0))
    fig.update_xaxes(tickprefix='GW', dtick=1)
    return fig
