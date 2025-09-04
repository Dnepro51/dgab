import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from .utils.confints import confint_group_statistic


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
        significance_level
    ):
    """Route to appropriate statistical test based on test_config."""
    pass


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
    
    return {
        'unique_grps_cnt': unique_grps_cnt,
        'test_config': test_config
    }