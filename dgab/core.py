import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from IPython.display import HTML
from .utils.confints import confint_group_statistic, confint_difference
from .utils.stat_tests import welch_ttest, anova_test, pairwise_tests_with_correction, chi2_test
from .utils.visualizations import plot_discrete, plot_binary_agg
from .utils.reports import generate_html_report, build_comprehensive_table
from .utils.validations import validate_inputs
from .utils.transformations import aggregate_to_individual_binary


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
def display_test_info(data_type, unique_grps_cnt, test_config, significance_level, confidence_level, dataframe, group_col, metric_col, statistic, dependency):
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
    print(f"Уровень значимости: {significance_level}")
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
        confidence_level,
        data_type,
        statistic,
        dependency
    ):
    display_test_info(data_type, unique_grps_cnt, test_config, significance_level, confidence_level, dataframe, group_col, metric_col, statistic, dependency)
    print()
    
    confint_method = test_config['confint_method']['statistic_value']
    confint_params = test_config['confint_params']['statistic_value']
    
    group_stats_df = confint_group_statistic(
        dataframe, group_col, metric_col, data_type, statistic,
        confint_method, confint_params, significance_level, confidence_level
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
        confidence_level,
        data_type,
        statistic
    ):
    """Route to appropriate statistical test based on test_config."""
    print("Результаты статистических тестов:")
    print()
    
    groups = sorted(dataframe[group_col].unique())
    
    omnibus_result = None
    omnibus_test = test_config['omnibus_test']
    if omnibus_test:
        omnibus_func = globals()[f"{omnibus_test}_test"]
        omnibus_result = omnibus_func(dataframe, group_col, metric_col, significance_level)
        omnibus_result['test_name'] = omnibus_test
        
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
        significance_level, confidence_level
    )
    
    diff_df = confint_difference(
        dataframe, group_col, metric_col, data_type, statistic,
        test_config['confint_method']['difference'],
        test_config['confint_params']['difference'],
        significance_level, confidence_level
    )
    
    pairwise_df = pairwise_tests_with_correction(
        dataframe, group_col, metric_col, test_func,
        correction_method, significance_level
    )
    
    print("Попарные сравнения:")
    display(pairwise_df)
    print()
    
    comprehensive_results = build_comprehensive_table(group_stats_df, diff_df, pairwise_df, statistic, significance_level, confidence_level)
    
    print("Сводная таблица результатов:")
    print(f"Сортировка: significant desc, group1_{statistic} desc, abs_difference asc")
    display(comprehensive_results)
    print()
    
    return group_stats_df, comprehensive_results, omnibus_result


# Функция запуска анализа

def how(data_type=None):
    """Show how to prepare data and use analyze() function for specific data_type."""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    implemented_types = ['discrete']
    available_types = [dt for dt in methods_route.keys() if dt in implemented_types]
    
    if data_type is None:
        raise ValueError(f"Укажите data_type. Доступные типы: {available_types}. dgab.how(data_type='discrete')")
    
    if data_type not in available_types:
        raise ValueError(f"Неизвестный data_type: {data_type}. Доступные: {available_types}")
    
    examples_path = os.path.join(os.path.dirname(__file__), 'data_examples.json')
    with open(examples_path, 'r') as f:
        examples = json.load(f)
    
    example_data = examples[data_type]
    
    print(f"Тип данных: {data_type}")
    print(f"Описание: {example_data['description']}")
    print()
    print("Пример данных (10 строк):")
    
    sample_df = pd.DataFrame(example_data['sample_data'])
    display(sample_df)
    print()
    
    print("Описание полей тестовых данных:")
    for field, desc in example_data['field_descriptions'].items():
        print(f"- {field}: {desc}")
    print()
    
    print(f"Параметры функции analyze(data_type='{data_type}'):")
    for param, info in example_data['parameters'].items():
        required_text = "обязательно" if info['required'] else "опционально"
        default_text = f", по умолчанию: {info['default']}" if info['default'] is not None else ""
        available_text = f", доступные значения: {info['available_values']}" if info['available_values'] is not None else ""
        
        print(f"- {param} ({info['type']}, {required_text}{default_text}{available_text})")
        print(f"  {info['description']}")
    print()
    
    print("Пример вызова analyze():")
    call_params = example_data['example_call']
    print("dgab.analyze(")
    for param, value in call_params.items():
        if isinstance(value, str) and param != 'dataframe':
            print(f"    {param}='{value}',")
        else:
            print(f"    {param}={value},")
    print(")")


def analyze(
        dataframe,
        data_type,
        group_col,
        metric_col,
        statistic='mean',
        dependency='independent',
        significance_level=0.01,
        confidence_level=0.99,
        metric_config=None
    ):
    validate_inputs(dataframe, data_type, group_col, metric_col, statistic, dependency, significance_level, metric_config)

    # Transform binary aggregated data to individual observations
    if data_type == 'binary_agg':
        dataframe = aggregate_to_individual_binary(dataframe, group_col, metric_config)
        metric_col = 'binary_outcome'  # Update metric column to transformed data

    unique_grps_cnt = count_groups(dataframe, group_col)
    test_config = get_test_config(data_type, unique_grps_cnt, statistic, dependency)
    
    fig = run_eda_analysis(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, confidence_level, data_type, statistic, dependency)
    
    group_stats_df, comprehensive_results, omnibus_result = run_statistical_test(dataframe, test_config, group_col, metric_col, unique_grps_cnt, significance_level, confidence_level, data_type, statistic)
    
    html_report = generate_html_report(group_stats_df, comprehensive_results, data_type, statistic, significance_level, confidence_level, unique_grps_cnt, omnibus_result=omnibus_result)
    display(HTML(html_report))