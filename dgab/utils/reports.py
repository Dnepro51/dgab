def format_number(value):
    """Clean formatting for numeric values."""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)

def format_ci(ci_list):
    """Clean formatting for confidence intervals."""
    if isinstance(ci_list, list) and len(ci_list) == 2:
        lower = float(ci_list[0])
        upper = float(ci_list[1])
        return f"[{lower:.4f}, {upper:.4f}]"
    return str(ci_list)

def format_count(count):
    """Format sample size with comma separators."""
    return f"{int(count):,}"

def generate_html_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level):
    """Generate HTML report summary for A/B test results."""
    confidence_level = int((1 - significance_level) * 100)
    
    group_stats_sorted = group_stats_df.sort_values(statistic, ascending=False).copy()
    best_group = group_stats_sorted.iloc[0]
    
    top_2_groups = group_stats_sorted.head(2)
    comparison = None
    if len(top_2_groups) > 1 and not comprehensive_results.empty:
        comp_filter = (
            ((comprehensive_results['group1'] == top_2_groups.iloc[1]['group']) &
             (comprehensive_results['group2'] == top_2_groups.iloc[0]['group'])) |
            ((comprehensive_results['group1'] == top_2_groups.iloc[0]['group']) &
             (comprehensive_results['group2'] == top_2_groups.iloc[1]['group']))
        )
        comp_matches = comprehensive_results[comp_filter]
        if not comp_matches.empty:
            comparison = comp_matches.iloc[0]
    
    confluence_css = """
    <style>
        .ab-report {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
            max-width: 700px;
            margin: 0;
            padding: 0;
            background-color: #fff;
            color: #172b4d;
        }
        .ab-report h3 {
            font-size: 20px;
            font-weight: 500;
            margin: 0 0 16px 0;
            color: #172b4d;
        }
        .ab-report h4 {
            font-size: 16px;
            font-weight: 500;
            margin: 16px 0 8px 0;
            color: #42526e;
        }
        .ab-report table {
            border-collapse: collapse;
            width: 100%;
            margin: 8px 0 16px 0;
            font-size: 14px;
        }
        .ab-report th {
            background-color: #f4f5f7;
            border: 1px solid #dfe1e6;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            color: #172b4d;
        }
        .ab-report td {
            border: 1px solid #dfe1e6;
            padding: 8px 12px;
            background-color: #fff;
        }
        .ab-report .number {
            text-align: right;
        }
        .ab-report .center {
            text-align: center;
        }
        .ab-report p {
            margin: 4px 0;
            font-size: 14px;
            line-height: 1.4;
        }
        .ab-report .group-name {
            font-weight: 600;
        }
    </style>
    """
    
    html = confluence_css + f"""
    <div class="ab-report">
        <h3>📊 A/B тест</h3>
        
        <h4>Группы теста:</h4>
        <table>
            <tr>
                <th>Группа</th>
                <th class="number">Размер выборки</th>
                <th class="center">{statistic.title()} (CI {confidence_level}%)</th>
                <th class="number">{statistic.title()}</th>
            </tr>
    """
    
    ci_col = f'ci_{confidence_level}'
    for _, row in group_stats_sorted.iterrows():
        html += f"""
            <tr>
                <td class="group-name">{row['group']}</td>
                <td class="number">{format_count(row['count'])}</td>
                <td class="center">{format_ci(row[ci_col])}</td>
                <td class="number">{format_number(row[statistic])}</td>
            </tr>
        """
    
    html += f"""
        </table>
        
        <h4>Результаты теста:</h4>
        <p><strong>Лучшая группа:</strong> {best_group['group']} ({format_number(best_group[statistic])})</p>
    """
    
    if comparison is not None:
        effect_ci_col = f'abs_difference_ci_{confidence_level}'
        html += f"""
        <p><strong>Различия значимы:</strong> {'Да' if comparison['significant'] else 'Нет'}</p>
        <p><strong>Размер эффекта:</strong> {format_number(comparison['abs_difference'])}</p>
        <p><strong>Доверительный интервал эффекта:</strong> {format_ci(comparison[effect_ci_col])}</p>
        """
    
    html += "</div>"
    return html