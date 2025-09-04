import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms


def t_ci(data, significance_level=0.01, **kwargs):
    """T-distribution confidence interval for mean.
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
    confidence_level = 1 - significance_level
    mean = np.mean(data)
    sem = stats.sem(data)
    df = len(data) - 1
    ci = stats.t.interval(confidence_level, df, loc=mean, scale=sem)
    return ci[0], ci[1]


def welch_ci(group1_data, group2_data, significance_level=0.01, **kwargs):
    """Welch's confidence interval for difference of means (unequal variances).
    
    https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.html
    https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.tconfint_diff.html#statsmodels.stats.weightstats.CompareMeans.tconfint_diff
    """
    cm = sms.CompareMeans(sms.DescrStatsW(group1_data), sms.DescrStatsW(group2_data))
    ci = cm.tconfint_diff(alpha=significance_level, usevar='unequal')
    return ci[0], ci[1]


def confint_group_statistic(dataframe, group_col, metric_col, data_type, statistic, 
                           confint_method, confint_params, significance_level=0.01):
    """Calculate confidence intervals for group statistics."""
    method_func = globals()[confint_method]
    results = []
    
    for group in dataframe[group_col].unique():
        group_data = dataframe[dataframe[group_col] == group][metric_col]
        
        if statistic == 'mean':
            stat_value = group_data.mean()
        
        ci_lower, ci_upper = method_func(group_data, significance_level=significance_level, **confint_params)
        
        results.append({
            'group': group,
            statistic: stat_value,
            'ci': [np.around(ci_lower, 4), np.around(ci_upper, 4)]
        })
    
    return pd.DataFrame(results)


def confint_difference(dataframe, group_col, metric_col, data_type, statistic,
                      confint_method, confint_params, significance_level=0.01):
    """Calculate confidence intervals for differences between groups."""
    method_func = globals()[confint_method]
    groups = sorted(dataframe[group_col].unique())
    results = []
    
    if len(groups) == 2:
        group1, group2 = groups
        group1_data = dataframe[dataframe[group_col] == group1][metric_col]
        group2_data = dataframe[dataframe[group_col] == group2][metric_col]
        
        if statistic == 'mean':
            difference = group2_data.mean() - group1_data.mean()
        
        ci_lower, ci_upper = method_func(group1_data, group2_data, 
                                       significance_level=significance_level, 
                                       **confint_params)
        
        results.append({
            'group1': group1,
            'group2': group2,
            'difference': difference,
            'ci': [np.around(ci_lower, 4), np.around(ci_upper, 4)]
        })
    
    else:
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                group1, group2 = groups[i], groups[j]
                group1_data = dataframe[dataframe[group_col] == group1][metric_col]
                group2_data = dataframe[dataframe[group_col] == group2][metric_col]
                
                if statistic == 'mean':
                    difference = group2_data.mean() - group1_data.mean()
                
                ci_lower, ci_upper = method_func(group1_data, group2_data,
                                               significance_level=significance_level,
                                               **confint_params)
                
                results.append({
                    'group1': group1,
                    'group2': group2,
                    'difference': difference,
                    'ci': [np.around(ci_lower, 4), np.around(ci_upper, 4)]
                })
    
    return pd.DataFrame(results)