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


def plot_binary_agg(dataframe, group_col, metric_col, **kwargs):
    """
    Binary aggregated data visualization with stacked bar chart.

    Input: Transformed individual binary data (0/1 values)
    Process: Reconstruct aggregated format internally
    Output: Stacked bar chart showing conversion rates by group
    """

    # Reconstruct aggregated data from transformed individual binary data
    groups_data = []
    for group in sorted(dataframe[group_col].unique()):
        group_data = dataframe[dataframe[group_col] == group][metric_col]
        total_users = len(group_data)
        conversions = sum(group_data)  # Sum of 1s = conversions
        failures = total_users - conversions
        groups_data.append({
            'group': group,
            'users': total_users,
            'conversions': conversions,
            'failures': failures
        })

    # Extract data for visualization
    groups = [item['group'] for item in groups_data]
    users = [item['users'] for item in groups_data]
    conversions = [item['conversions'] for item in groups_data]
    failures = [item['failures'] for item in groups_data]

    # Calculate proportions
    conv_proportions = [c / u for c, u in zip(conversions, users)]
    fail_proportions = [f / u for f, u in zip(failures, users)]

    colors = px.colors.qualitative.Dark24[:len(groups)]

    fig = go.Figure()

    for i, (group, conv, fail, total, conv_prop, fail_prop, color) in enumerate(zip(
            groups, conversions, failures, users, conv_proportions, fail_proportions, colors)):

        fig.add_trace(go.Bar(
            x=[group],
            y=[conv_prop],
            name=f'Группа {group} - Конверсии',
            marker_color=color,
            opacity=0.8,
            text=f'{conv}<br>({conv_prop:.1%})',
            textposition='inside',
            textfont=dict(size=12, color='black', family='Arial Black'),
            showlegend=False
        ))

        fig.add_trace(go.Bar(
            x=[group],
            y=[fail_prop],
            name=f'Группа {group} - Не конверсии',
            marker_color=color,
            opacity=0.35,
            text=f'{fail}<br>({fail_prop:.1%})',
            textposition='inside',
            textfont=dict(size=12, color='black', family='Arial Black'),
            showlegend=False
        ))

    for i, (group, total) in enumerate(zip(groups, users)):
        fig.add_annotation(
            x=group,
            y=1.05,
            text=f'{total}',
            showarrow=False,
            font=dict(size=12, weight='bold', color='black')
        )

    fig.update_layout(
        barmode='stack',
        title='Анализ конверсий по группам',
        xaxis_title=group_col,
        yaxis_title='Пропорция пользователей',
        template='plotly_white'
    )

    fig.update_yaxes(range=[0, 1.1])

    return fig