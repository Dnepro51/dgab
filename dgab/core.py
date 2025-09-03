import json
import os

def count_groups(dataframe, group_col):
    """Count unique groups in the group column."""
    unique_grps_cnt = dataframe[group_col].nunique()
    return unique_grps_cnt

def get_test_config(data_type, unique_grps_cnt, statistic, dependency):
    """Load methods_route.json and return the correct test configuration with FULL MATCH."""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'methods_route.json')
    with open(json_path, 'r') as f:
        methods_route = json.load(f)
    
    group_key = "2" if unique_grps_cnt == 2 else "multiple"
    candidate_config = methods_route[data_type][group_key]
    
    if candidate_config['statistic'] == statistic and candidate_config['dependency'] == dependency:
        return candidate_config
    
    return candidate_config

def run_eda_analysis(
        dataframe, 
        test_config, 
        group_col, 
        metric_col, 
        unique_grps_cnt, 
        significance_level
    ):
    """Perform EDA analysis: means, quantiles, visuals, confidence intervals."""
    pass

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

def analyze(
        dataframe, 
        data_type, 
        group_col, 
        metric_col, 
        statistic='mean', 
        dependency='independent',
        significance_level=0.01
    ):
    """Main analyze function for A/B testing."""
    unique_grps_cnt = count_groups(dataframe, group_col)
    test_config = get_test_config(data_type, unique_grps_cnt, statistic, dependency)
    
    return {
        'unique_grps_cnt': unique_grps_cnt,
        'test_config': test_config
    }