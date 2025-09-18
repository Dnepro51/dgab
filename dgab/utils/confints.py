import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms


def t_ci(data, significance_level=0.01, confidence_level=0.99, **kwargs):
    """T-distribution confidence interval for mean.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    df = len(data) - 1
    ci = stats.t.interval(confidence_level, df, loc=mean, scale=sem)
    return ci[0], ci[1]


def welch_ci(group1_data, group2_data, significance_level=0.01, confidence_level=0.99, **kwargs):
    """Welch's confidence interval for difference of means (unequal variances).

    https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.html
    https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.tconfint_diff.html#statsmodels.stats.weightstats.CompareMeans.tconfint_diff
    """
    cm = sms.CompareMeans(sms.DescrStatsW(group1_data), sms.DescrStatsW(group2_data))
    alpha = 1 - confidence_level
    ci = cm.tconfint_diff(alpha=alpha, usevar='unequal')
    return ci[0], ci[1]


def confint_group_statistic(dataframe, group_col, metric_col, data_type, statistic,
                           confint_method, confint_params, significance_level=0.01, confidence_level=0.99):
    """Calculate confidence intervals for group statistics."""
    method_func = globals()[confint_method]
    results = []

    confidence_level_int = int(confidence_level * 100)
    ci_column_name = f'ci_{confidence_level_int}'

    for group in dataframe[group_col].unique():
        group_data = dataframe[dataframe[group_col] == group][metric_col]

        if statistic == 'mean' or statistic == 'proportion':
            stat_value = group_data.mean()

        ci_lower, ci_upper = method_func(group_data, significance_level=significance_level, confidence_level=confidence_level, **confint_params)

        result = {
            'group': group,
            'count': len(group_data),
            statistic: stat_value,
            ci_column_name: [np.around(ci_lower, 4), np.around(ci_upper, 4)]
        }

        # For binary_agg, add trials and successes columns
        if data_type == 'binary_agg':
            trials = len(group_data)
            successes = int(sum(group_data))
            result['trials'] = trials
            result['successes'] = successes

        results.append(result)

    return pd.DataFrame(results)


def confint_difference(dataframe, group_col, metric_col, data_type, statistic,
                      confint_method, confint_params, significance_level=0.01, confidence_level=0.99):
    """Calculate confidence intervals for differences between groups."""
    method_func = globals()[confint_method]
    groups = sorted(dataframe[group_col].unique())
    results = []

    confidence_level_int = int(confidence_level * 100)
    ci_column_name = f'ci_{confidence_level_int}'
    
    if len(groups) == 2:
        group1, group2 = groups
        group1_data = dataframe[dataframe[group_col] == group1][metric_col]
        group2_data = dataframe[dataframe[group_col] == group2][metric_col]
        
        if statistic == 'mean' or statistic == 'proportion':
            difference = group2_data.mean() - group1_data.mean()
        
        ci_lower, ci_upper = method_func(group1_data, group2_data,
                                       significance_level=significance_level,
                                       confidence_level=confidence_level,
                                       **confint_params)
        
        results.append({
            'group1': group1,
            'group2': group2,
            'difference': difference,
            ci_column_name: [np.around(ci_lower, 4), np.around(ci_upper, 4)]
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
                    ci_column_name: [np.around(ci_lower, 4), np.around(ci_upper, 4)]
                })

    return pd.DataFrame(results)


def wilson_ci(data, significance_level=0.01, confidence_level=0.99, **kwargs):
    """Wilson confidence interval for proportions.

    https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
    """
    from statsmodels.stats.proportion import proportion_confint

    successes = sum(data)
    trials = len(data)
    alpha = 1 - confidence_level

    ci_lower, ci_upper = proportion_confint(successes, trials, alpha=alpha, method='wilson')
    return ci_lower, ci_upper


def newcombe_wilson_ci(group1_data, group2_data, significance_level=0.01, confidence_level=0.99, **kwargs):
    """Newcombe-Wilson confidence interval for difference between proportions.

    https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html
    """
    from statsmodels.stats.proportion import confint_proportions_2indep

    count1, nobs1 = sum(group1_data), len(group1_data)
    count2, nobs2 = sum(group2_data), len(group2_data)
    alpha = 1 - confidence_level

    ci_lower, ci_upper = confint_proportions_2indep(count1, nobs1, count2, nobs2,
                                                  method='newcombe', alpha=alpha)
    return ci_lower, ci_upper