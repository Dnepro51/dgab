import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from .utils.confints import confint_group_statistic, confint_difference
from .utils.stat_tests import welch_ttest, anova_test, pairwise_tests_with_correction
from .utils.visualizations import plot_discrete


# Утилиты для определения конфигурации теста

def count_groups(dataframe, group_col):
    """Count unique groups in the group column."""
    unique_grps_cnt = dataframe[group_col].nunique()
    return unique_grps_cnt

def get_test_config(data_type, unique_grps_cnt, statistic, dependency):
    json_path = os.path.join(os.path.dirname(__file__), '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    group_key = "2" if unique_grps_cnt == 2 else "multiple"
    test_config = methods_route[data_type][group_key][statistic][dependency]
    
    return test_config


# EDA-функции

## EDA-1 Отображение информации о конфигурации теста
def display_test_info(data_type, unique_grps_cnt, test_config, significance_level, dataframe, group_col, metric_col, statistic, dependency):
    data_type_ru = {'discrete': 'дискретные', 'binary_agg': 'бинарные', 'continuous': 'непрерывные'}
    test_name_ru = {'welch_ttest': 'T-тест Уэлча', 'anova': 'ANOVA', 'chi2': 'Хи-квадрат'}
    correction_ru = {'bonferroni': 'Бонферрони', None: 'нет'}
    dependency_ru = {'independent': 'независимые', 'dependent': 'зависимые'}
    confint_method_ru = {
        't_ci': 'T-распределение',
        'welch_ci': 'Уэлча',
        'wilson_ci': 'Уилсона',
        'newcombe_wilson_ci': 'Ньюкомба-Уилсона',
        None: 'нет'
    }
    
    confidence_level = 1 - significance_level
    
    print(f"Тип данных: {data_type_ru.get(data_type, data_type)}")
    print(f"Статистика: {statistic}")
    print(f"Групп: {unique_grps_cnt}")
    print(f"Значимость: {significance_level}")
    print(f"Доверительная вероятность: {confidence_level}")
    print(f"Зависимость выборок: {dependency_ru.get(dependency, dependency)}")
    print(f"Колонка с идентификатором групп: {group_col}")
    print(f"Колонка с метрикой: {metric_col}")
    
    group_names = sorted(dataframe[group_col].unique())
    print(f"Названия групп: {group_names}")
    
    if unique_grps_cnt == 2:
        print(f"Тест: {test_name_ru.get(test_config['test_name'], test_config['test_name'])}")
    else:
        if test_config['omnibus_test']:
            print(f"Общий тест: {test_name_ru.get(test_config['omnibus_test'], test_config['omnibus_test'])}")
            print(f"Тесты для попарных сравнений: {test_name_ru.get(test_config['test_name'], test_config['test_name'])}")
        print(f"Коррекция: {correction_ru.get(test_config['multiple_comparison_correction'])}")
    # Добавляем вывод методов доверительных интервалов
    confint_methods = test_config.get('confint_method', {})
    statistic_method = confint_methods.get('statistic_value')
    difference_method = confint_methods.get('difference')
    
    print(f"Доверительный интервал для {statistic}: {confint_method_ru.get(statistic_method, statistic_method)}")
    print(f"Доверительный интервал для differences: {confint_method_ru.get(difference_method, difference_method)}")





def run_eda_analysis(
        dataframe, 
        test_config, 
        group_col, 
        metric_col, 
        unique_grps_cnt, 
        significance_level,
        data_type,
        statistic,
        dependency
    ):
    display_test_info(data_type, unique_grps_cnt, test_config, significance_level, dataframe, group_col, metric_col, statistic, dependency)
    print()
    
    confint_method = test_config['confint_method']['statistic_value']
    confint_params = test_config['confint_params']['statistic_value']
    
    group_stats_df = confint_group_statistic(
        dataframe, group_col, metric_col, data_type, statistic,
        confint_method, confint_params, significance_level
    )
    group_stats_df = group_stats_df.sort_values(statistic, ascending=False)
    
    print("Статистика по группам:")
    display(group_stats_df)
    print()
    
    viz_function = globals()[test_config['visualization_function']]
    fig = viz_function(dataframe, group_col, metric_col)
    fig.show()
    print()
    
    return fig




# Стат-тест функции

def run_statistical_test(
        dataframe, 
        test_config, 
        group_col, 
        metric_col, 
        unique_grps_cnt, 
        significance_level,
        data_type,
        statistic
    ):
    """Route to appropriate statistical test based on test_config."""
    print("Результаты статистических тестов:")
    print()
    
    groups = sorted(dataframe[group_col].unique())
    
    omnibus_test = test_config['omnibus_test']
    if omnibus_test:
        omnibus_func = globals()[f"{omnibus_test}_test"]
        omnibus_result = omnibus_func(dataframe, group_col, metric_col, significance_level)
        
        print(f"Общий тест: {omnibus_test}")
        print(f"Статистика: {omnibus_result['statistic']:.4f}")
        print(f"P-value: {omnibus_result['pvalue']:.6f}")
        print(f"Значимый: {'Да' if omnibus_result['significant'] else 'Нет'}")
        print()
    
    test_func = globals()[test_config['test_name']]
    correction_method = test_config['multiple_comparison_correction']
    
    group_stats_df = confint_group_statistic(
        dataframe, group_col, metric_col, data_type, statistic,
        test_config['confint_method']['statistic_value'],
        test_config['confint_params']['statistic_value'],
        significance_level
    )
    
    diff_df = confint_difference(
        dataframe, group_col, metric_col, data_type, statistic,
        test_config['confint_method']['difference'],
        test_config['confint_params']['difference'],
        significance_level
    )
    
    pairwise_df = pairwise_tests_with_correction(
        dataframe, group_col, metric_col, test_func,
        correction_method, significance_level
    )
    
    print("Попарные сравнения:")
    display(pairwise_df)
    print()
    
    results = []
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
        
        confidence_level_int = int((1 - significance_level) * 100)
        ci_col = f'ci_{confidence_level_int}'
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
    
    print("Сводная таблица результатов:")
    display(comprehensive_results)
    print()


# Функция запуска анализа

def analyze(
        dataframe, 
        data_type, 
        group_col, 
        metric_col, 
        statistic='mean', 
        dependency='independent',
        significance_level=0.01
    ):
    unique_grps_cnt = count_groups(dataframe, group_col)
    test_config = get_test_config(data_type, unique_grps_cnt, statistic, dependency)
    
    fig = run_eda_analysis(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, data_type, statistic, dependency)
    
    run_statistical_test(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, data_type, statistic)
    
    return {
        'unique_grps_cnt': unique_grps_cnt,
        'test_config': test_config,
        'visualization': fig
    }