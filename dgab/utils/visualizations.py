import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_discrete(dataframe, group_col, metric_col, bins=None):
    groups = sorted(dataframe[group_col].unique())  
    colors = px.colors.qualitative.Dark24[:len(groups)]
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    for i, group in enumerate(groups):
        group_data = dataframe[dataframe[group_col] == group][metric_col]
        
        histogram_kwargs = {
            'x': group_data,
            'name': f'Группа {group}',
            'legendgroup': f'group_{group}',
            'histnorm': 'probability',
            'marker_color': colors[i],
            'opacity': 0.35,
            'showlegend': True
        }
        
        if bins is not None:
            histogram_kwargs['nbinsx'] = bins
            
        fig.add_trace(go.Histogram(**histogram_kwargs), row=1, col=1)
    
    max_sample_size = 5000
    for i, group in enumerate(reversed(groups)):
        idx = groups.index(group)
        group_data = dataframe[dataframe[group_col] == group][metric_col]
        
        if len(group_data) > max_sample_size:
            group_data = group_data.sample(n=max_sample_size, random_state=42)
        
        fig.add_trace(go.Box(
            x=group_data,
            name=f'Группа {group}',
            legendgroup=f'group_{group}',
            marker_color=colors[idx],
            opacity=0.35,
            showlegend=False,
            x0=0,
            alignmentgroup=True
        ), row=2, col=1)
    
    fig.update_xaxes(title_text=metric_col, row=2, col=1)
    fig.update_yaxes(title_text="Вероятность", row=1, col=1)
    
    all_data = dataframe[metric_col]
    x_min = all_data.min()
    x_max = all_data.max()
    x_range_start = x_min - 0.1 if x_min == 0 else x_min - 0.5
    x_range = [x_range_start, x_max + 0.5]
    
    fig.update_xaxes(
        range=x_range, 
        autorange=False,
        showticklabels=True,
        tick0=x_min,
        dtick=1,
        row=1, col=1
    )
    fig.update_xaxes(
        range=x_range, 
        autorange=False,
        showticklabels=True,
        tick0=x_min,
        dtick=1,
        row=2, col=1
    )
    
    fig.update_xaxes(domain=[0.01, 1], row=1, col=1)
    fig.update_xaxes(domain=[0.0, 1], row=2, col=1)
    
    fig.update_layout(
            title=f'Анализ распределения {metric_col} по группам',
            template='plotly_white',
            barmode='overlay',
            boxgap=0.3,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    return fig


def plot_binary_agg(dataframe, group_col, metric_col):
    """Temporary placeholder for binary_agg visualization - just returns empty figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text="plot_binary_agg() placeholder - visualization not implemented yet",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig.update_layout(title="Binary Aggregated Data Visualization (Coming Soon)")

    return fig