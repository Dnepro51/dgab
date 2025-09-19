def format_number(value):
    """Clean formatting for numeric values."""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def get_statistic_russian(statistic):
    """Get Russian name for statistic."""
    statistic_ru = {
        'mean': 'среднее',
        'median': 'медиана',
        'proportion': 'пропорция'
    }
    return statistic_ru.get(statistic, statistic)

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

def build_comprehensive_table(group_stats_df, diff_df, pairwise_df, statistic, significance_level, confidence_level=0.99):
    """Build comprehensive results table universal for all data types."""
    import pandas as pd
    import numpy as np

    results = []
    confidence_level_int = int(confidence_level * 100)
    ci_col = f'ci_{confidence_level_int}'
    
    for i, row in pairwise_df.iterrows():
        group1, group2 = row['group1'], row['group2']
        
        group1_stats = group_stats_df[group_stats_df['group'] == group1].iloc[0]
        group2_stats = group_stats_df[group_stats_df['group'] == group2].iloc[0]
        diff_row = diff_df[(diff_df['group1'] == group1) & (diff_df['group2'] == group2)]
        
        if len(diff_row) == 0:
            diff_row = diff_df[(diff_df['group1'] == group2) & (diff_df['group2'] == group1)]
            difference = -diff_row.iloc[0]['difference'] if len(diff_row) > 0 else 0
        else:
            difference = diff_row.iloc[0]['difference']
        
        abs_difference = abs(difference)
        comparison_result = f"{group1}>{group2}" if group1_stats[statistic] > group2_stats[statistic] else f"{group2}>{group1}"
        
        diff_ci = diff_row.iloc[0][ci_col] if len(diff_row) > 0 else [0, 0]
        abs_diff_ci = [abs(diff_ci[0]), abs(diff_ci[1])]
        abs_diff_ci.sort()
        
        results.append({
            'group1': group1,
            'group1_count': group1_stats['count'],
            f'group1_{statistic}': np.around(group1_stats[statistic], 4),
            f'group1_{ci_col}': group1_stats[ci_col],
            'group2': group2,
            'group2_count': group2_stats['count'],
            f'group2_{statistic}': np.around(group2_stats[statistic], 4),
            f'group2_{ci_col}': group2_stats[ci_col],
            'abs_difference': np.around(abs_difference, 4),
            f'abs_difference_{ci_col}': [np.around(abs_diff_ci[0], 4), np.around(abs_diff_ci[1], 4)],
            'comparison_result': comparison_result,
            'pvalue': row.get('pvalue', 0),
            'corrected_pvalue': row.get('corrected_pvalue', None),
            'significant': row.get('significant', False)
        })
    
    comprehensive_results = pd.DataFrame(results)
    comprehensive_results = comprehensive_results.sort_values(['significant', f'group1_{statistic}', 'abs_difference'], ascending=[False, False, True])
    
    return comprehensive_results


def generate_confluence_css():
    """Generate CSS for professional Confluence-style reports."""
    return """
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
        .ab-report .significant-yes {
            color: #00875a;
            font-weight: 600;
        }
        .ab-report .significant-no {
            color: #de350b;
            font-weight: 600;
        }
    </style>
    """


def generate_group_stats_table(group_stats_df, statistic, significance_level, confidence_level=0.99):
    """Generate group statistics table HTML."""
    confidence_level_int = int(confidence_level * 100)
    group_stats_sorted = group_stats_df.sort_values(statistic, ascending=False).copy()
    ci_col = f'ci_{confidence_level_int}'
    
    html = f"""
        <h4>Группы теста:</h4>
        <table>
            <tr>
                <th>Группа</th>
                <th class="number">Размер выборки</th>
                <th class="center">{statistic.title()} (CI {confidence_level_int}%)</th>
                <th class="number">{statistic.title()}</th>
            </tr>
    """
    
    for _, row in group_stats_sorted.iterrows():
        html += f"""
            <tr>
                <td class="group-name">{row['group']}</td>
                <td class="number">{format_count(row['count'])}</td>
                <td class="center">{format_ci(row[ci_col])}</td>
                <td class="number">{format_number(row[statistic])}</td>
            </tr>
        """
    
    html += "</table>"
    return html, group_stats_sorted


def generate_2group_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level=0.99, omnibus_result=None, fig=None):
    """Generate HTML report for 2-group A/B test."""
    confidence_level_int = int(confidence_level * 100)

    group_stats_table, group_stats_sorted = generate_group_stats_table(group_stats_df, statistic, significance_level, confidence_level)
    best_group = group_stats_sorted.iloc[0]
    
    comparison = comprehensive_results.iloc[0] if not comprehensive_results.empty else None
    
    html = generate_confluence_css() + f"""
    <div class="ab-report">
        <h3>📊 A/B тест (2 группы)</h3>
        
        {group_stats_table}
        
        <h4>Результаты теста:</h4>
        <p><strong>Исследуемая статистика:</strong> {get_statistic_russian(statistic)}</p>
        <p><strong>Лучшая группа:</strong> {best_group['group']} ({format_number(best_group[statistic])})</p>
    """
    
    if comparison is not None:
        effect_ci_col = f'abs_difference_ci_{confidence_level_int}'
        html += f"""
        <p><strong>Различия значимы:</strong> {'Да' if comparison['significant'] else 'Нет'}</p>
        <p><strong>Размер эффекта:</strong> {format_number(comparison['abs_difference'])}</p>
        <p><strong>Доверительный интервал эффекта:</strong> {format_ci(comparison[effect_ci_col])}</p>
        """
    
    html += "</div>"
    return html


def generate_multigroup_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level=0.99, omnibus_result=None, fig=None):
    """Generate HTML report for multi-group A/B test."""
    confidence_level_int = int(confidence_level * 100)

    group_stats_table, group_stats_sorted = generate_group_stats_table(group_stats_df, statistic, significance_level, confidence_level)
    best_group = group_stats_sorted.iloc[0]
    
    html = generate_confluence_css() + f"""
    <div class="ab-report">
        <h3>📊 A/B тест (множественные группы)</h3>
        
        {group_stats_table}
        
        <h4>Результаты теста:</h4>
        <p><strong>Исследуемая статистика:</strong> {get_statistic_russian(statistic)}</p>
        <p><strong>Лучшая группа:</strong> {best_group['group']} ({format_number(best_group[statistic])})</p>
    """
    
    if omnibus_result:
        html += f"""
        <p><strong>Общие различия значимы:</strong> {'Да' if omnibus_result['significant'] else 'Нет'}</p>
        """
    
    html += f"""
        
        <h4>🔍 Попарные сравнения:</h4>
        <table>
            <tr>
                <th>Сравнение</th>
                <th class="number">{get_statistic_russian(statistic).capitalize()} левой группы</th>
                <th class="number">{get_statistic_russian(statistic).capitalize()} правой группы</th>
                <th class="center">Значимо</th>
                <th class="number">Размер эффекта</th>
                <th class="center">Доверительный интервал эффекта</th>
            </tr>
    """
    
    for _, row in comprehensive_results.iterrows():
        significant_class = "significant-yes" if row['significant'] else "significant-no"
        significant_text = "✅ Да" if row['significant'] else "❌ Нет"
        effect_ci_col = f'abs_difference_ci_{confidence_level_int}'
        
        group1_stat = row[f'group1_{statistic}']
        group2_stat = row[f'group2_{statistic}']
        if group1_stat > group2_stat:
            comparison_text = f"{row['group1']}>{row['group2']}"
            left_stat = group1_stat
            right_stat = group2_stat
        else:
            comparison_text = f"{row['group2']}>{row['group1']}"
            left_stat = group2_stat
            right_stat = group1_stat
        
        html += f"""
            <tr>
                <td class="group-name">{comparison_text}</td>
                <td class="number">{format_number(left_stat)}</td>
                <td class="number">{format_number(right_stat)}</td>
                <td class="center {significant_class}">{significant_text}</td>
                <td class="number">{format_number(row['abs_difference'])}</td>
                <td class="center">{format_ci(row[effect_ci_col])}</td>
            </tr>
        """
    
    html += "</table></div>"
    return html


def generate_html_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level, unique_grps_cnt, omnibus_result=None, fig=None):
    """Generate HTML report - routes to 2-group or multi-group version."""
    if unique_grps_cnt == 2:
        return generate_2group_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level, omnibus_result, fig)
    else:
        return generate_multigroup_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level, omnibus_result, fig)