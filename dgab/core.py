import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from .utils.confints import confint_group_statistic, confint_difference
from .utils.stat_tests import welch_ttest, anova_test, pairwise_tests_with_correction
from .utils.results import create_comprehensive_results


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
    
    print(f"Тип данных: {data_type_ru.get(data_type, data_type)}")
    print(f"Статистика: {statistic}")
    print(f"Групп: {unique_grps_cnt}")
    print(f"Значимость: {significance_level}")
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


## EDA-2 Сбор доверительных интервалов по методам из @methods_route.json

def statistic_confint_display(df, group_col, metric_col, data_type, statistic):
    
    


    pass


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
    
    confidence_level = 1 - significance_level
    print(f"Статистика по группам. Доверительный интервал для {statistic}: {confidence_level}")
    display(group_stats_df)
    print()




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
    
    if len(groups) == 2:
        group1_data = dataframe[dataframe[group_col] == groups[0]][metric_col]
        group2_data = dataframe[dataframe[group_col] == groups[1]][metric_col]
        
        result = test_func(group1_data, group2_data, significance_level)
        
        print(f"Тест: {test_config['test_name']}")
        print(f"Статистика: {result['statistic']:.4f}")
        print(f"P-value: {result['pvalue']:.6f}")
        print(f"Значимый: {'Да' if result['significant'] else 'Нет'}")
        print()
    else:
        pairwise_df = pairwise_tests_with_correction(
            dataframe, group_col, metric_col, test_func,
            correction_method, significance_level
        )
        
        print("Попарные сравнения:")
        display(pairwise_df)
        print()
    
    confint_method = test_config['confint_method']['difference']
    confint_params = test_config['confint_params']['difference']
    
    diff_df = confint_difference(
        dataframe, group_col, metric_col, data_type, statistic,
        confint_method, confint_params, significance_level
    )
    
    confidence_level = 1 - significance_level
    print(f"Доверительные интервалы для разностей {statistic}: {confidence_level}")
    display(diff_df)
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
    
    run_eda_analysis(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, data_type, statistic, dependency)
    
    run_statistical_test(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, data_type, statistic)
    
    return {
        'unique_grps_cnt': unique_grps_cnt,
        'test_config': test_config
    }