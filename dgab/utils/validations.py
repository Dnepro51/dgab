import json
import os
import pandas as pd
import numpy as np


def validate_dataframe(dataframe):
    """Validate that input is pandas DataFrame.
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Ожидается pandas DataFrame, получен {type(dataframe).__name__}")
    
    if dataframe.empty:
        raise ValueError("DataFrame пустой - нет данных для анализа")


def validate_required_columns(dataframe, group_col, metric_col, metric_config=None):
    """Validate that required columns exist in DataFrame.
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
    """
    columns = dataframe.columns.tolist()
    
    if group_col not in columns:
        raise ValueError(f"Колонка с группами '{group_col}' не найдена. Доступные колонки: {columns}")
    
    if metric_col not in columns:
        raise ValueError(f"Колонка с метрикой '{metric_col}' не найдена. Доступные колонки: {columns}")
    
    if metric_config:
        if 'trials_col_name' in metric_config and metric_config['trials_col_name'] not in columns:
            raise ValueError(f"Колонка trials '{metric_config['trials_col_name']}' не найдена. Доступные колонки: {columns}")
        
        if 'successes_col_name' in metric_config and metric_config['successes_col_name'] not in columns:
            raise ValueError(f"Колонка successes '{metric_config['successes_col_name']}' не найдена. Доступные колонки: {columns}")


def validate_metric_column_type(dataframe, metric_col, data_type):
    """Validate metric column contains appropriate data type.
    Note: discrete data can be both integers and floats.
    
    https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
    """
    if data_type in ['discrete', 'continuous']:
        if not pd.api.types.is_numeric_dtype(dataframe[metric_col]):
            raise ValueError(f"Колонка '{metric_col}' должна содержать численные данные (int или float) для типа '{data_type}', получен тип {dataframe[metric_col].dtype}")
        
        if dataframe[metric_col].isna().any():
            raise ValueError(f"Колонка '{metric_col}' содержит пропущенные значения (NaN)")


def validate_group_column(dataframe, group_col):
    """Validate group column has valid values.
    
    https://pandas.pydata.org/docs/reference/api/pandas.Series.nunique.html
    """
    if dataframe[group_col].isna().any():
        raise ValueError(f"Колонка с группами '{group_col}' содержит пропущенные значения (NaN)")
    
    unique_groups = dataframe[group_col].nunique()
    if unique_groups < 2:
        raise ValueError(f"Недостаточно групп для сравнения: {unique_groups}. Минимум 2 группы")
    
    if unique_groups > 10:
        raise ValueError(f"Слишком много групп: {unique_groups}. Максимум 10 групп")


def validate_parameters(data_type, statistic, dependency):
    """Validate parameter values against JSON configuration.
    
    https://docs.python.org/3/library/json.html
    """
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    available_data_types = list(methods_route.keys())
    if data_type not in available_data_types:
        raise ValueError(f"Неизвестный data_type: '{data_type}'. Доступные: {available_data_types}")
    
    available_statistics = []
    for group_count in methods_route[data_type].values():
        available_statistics.extend(group_count.keys())
    available_statistics = list(set(available_statistics))
    
    if statistic not in available_statistics:
        raise ValueError(f"Неизвестная статистика: '{statistic}'. Доступные для {data_type}: {available_statistics}")
    
    available_dependencies = []
    for group_count in methods_route[data_type].values():
        for stat_config in group_count.values():
            available_dependencies.extend(stat_config.keys())
    available_dependencies = list(set(available_dependencies))
    
    if dependency not in available_dependencies:
        raise ValueError(f"Неизвестная зависимость: '{dependency}'. Доступные для {data_type}: {available_dependencies}")


def validate_sample_sizes(dataframe, group_col, min_sample_size=1):
    """Validate that each group has at least 1 observation.
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
    """
    group_sizes = dataframe.groupby(group_col).size()
    empty_groups = group_sizes[group_sizes < min_sample_size]
    
    if len(empty_groups) > 0:
        empty_group_info = empty_groups.to_dict()
        raise ValueError(f"Пустые группы найдены: {empty_group_info}. Каждая группа должна содержать хотя бы 1 наблюдение")


def validate_config_requirements(data_type, metric_config, test_config):
    """Validate special configuration requirements.
    
    https://docs.python.org/3/library/json.html
    """
    if test_config.get('custom_config_required', False):
        if data_type == 'binary_agg':
            if not metric_config:
                raise ValueError("Для типа 'binary_agg' требуется параметр metric_config с 'trials_col_name' и 'successes_col_name'")
            
            required_keys = ['trials_col_name', 'successes_col_name']
            missing_keys = [key for key in required_keys if key not in metric_config]
            
            if missing_keys:
                raise ValueError(f"В metric_config отсутствуют обязательные ключи: {missing_keys}")


def validate_binary_agg_data(dataframe, metric_config):
    """Validate binary aggregated data constraints.
    
    https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_integer_dtype.html
    """
    trials_col = metric_config['trials_col_name']
    successes_col = metric_config['successes_col_name']
    
    if not pd.api.types.is_numeric_dtype(dataframe[trials_col]):
        raise ValueError(f"Колонка trials '{trials_col}' должна содержать численные данные")
    
    if not pd.api.types.is_numeric_dtype(dataframe[successes_col]):
        raise ValueError(f"Колонка successes '{successes_col}' должна содержать численные данные")
    
    if (dataframe[trials_col] < 0).any():
        raise ValueError(f"Колонка trials '{trials_col}' не может содержать отрицательные значения")
    
    if (dataframe[successes_col] < 0).any():
        raise ValueError(f"Колонка successes '{successes_col}' не может содержать отрицательные значения")
    
    if (dataframe[successes_col] > dataframe[trials_col]).any():
        raise ValueError(f"Количество успехов не может превышать количество попыток: successes <= trials")
    
    if (dataframe[trials_col] == 0).any():
        raise ValueError(f"Колонка trials '{trials_col}' не может содержать нулевые значения")


def validate_inputs(
        dataframe, 
        data_type, 
        group_col, 
        metric_col, 
        statistic='mean', 
        dependency='independent',
        significance_level=0.01,
        metric_config=None
    ):
    """Main validation orchestrator function.
    
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """
    validate_dataframe(dataframe)
    
    validate_required_columns(dataframe, group_col, metric_col, metric_config)
    
    validate_metric_column_type(dataframe, metric_col, data_type)
    
    validate_group_column(dataframe, group_col)
    
    validate_parameters(data_type, statistic, dependency)
    
    if significance_level <= 0 or significance_level >= 1:
        raise ValueError(f"Уровень значимости должен быть между 0 и 1, получен: {significance_level}")
    
    validate_sample_sizes(dataframe, group_col)
    
    unique_grps_cnt = dataframe[group_col].nunique()
    group_key = "2" if unique_grps_cnt == 2 else "multiple"
    
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    test_config = methods_route[data_type][group_key][statistic][dependency]
    
    validate_config_requirements(data_type, metric_config, test_config)
    
    if data_type == 'binary_agg' and metric_config:
        validate_binary_agg_data(dataframe, metric_config)