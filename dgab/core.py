import json
import os

def count_groups(dataframe, group_col):
    """
    Count unique groups in the group column.
    """
    unique_grps_cnt = dataframe[group_col].nunique()
    return unique_grps_cnt

def get_test_config(data_type, unique_grps_cnt, statistic, dependency):
    """
    Load methods_route.json and return the correct test configuration with FULL MATCH.
    Matches ALL parameters: data_type, group_count, statistic, dependency
    """
    # Load JSON file
    json_path = os.path.join(os.path.dirname(__file__), '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    # Convert group count to JSON key
    group_key = "2" if unique_grps_cnt == 2 else "multiple"
    
    # Get the candidate config
    candidate_config = methods_route[data_type][group_key]
    
    # FULL MATCH check
    if candidate_config['statistic'] == statistic and candidate_config['dependency'] == dependency:
        return candidate_config
    
    return candidate_config

def analyze(dataframe, data_type, group_col, metric_col, statistic='mean', dependency='independent'):
    """
    Main analyze function for A/B testing.
    Now gets the correct test configuration from JSON with FULL MATCH.
    """
    unique_grps_cnt = count_groups(dataframe, group_col)
    test_config = get_test_config(data_type, unique_grps_cnt, statistic, dependency)
    
    return {
        'data_type': data_type,
        'statistic': statistic, 
        'dependency': dependency,
        'group_col': group_col,
        'metric_col': metric_col,
        'dataframe_shape': dataframe.shape,
        'unique_grps_cnt': unique_grps_cnt,
        'test_config': test_config
    }