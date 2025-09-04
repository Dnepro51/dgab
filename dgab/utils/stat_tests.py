import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from . import corrections


def welch_ttest(group1_data, group2_data, significance_level=0.01):
    """Welch's t-test for two independent groups with unequal variances.
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    """
    statistic, pvalue = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    significant = pvalue < significance_level
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': significant
    }


def anova_test(dataframe, group_col, metric_col, significance_level=0.01):
    """One-way ANOVA test for multiple groups.
    
    https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    """
    groups = [group_data[metric_col].values for name, group_data in dataframe.groupby(group_col)]
    statistic, pvalue = stats.f_oneway(*groups)
    significant = pvalue < significance_level
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': significant
    }


def pairwise_tests_with_correction(dataframe, group_col, metric_col, test_func, 
                                  correction_method, significance_level=0.01):
    """Perform pairwise tests with multiple comparison correction."""
    groups = sorted(dataframe[group_col].unique())
    results = []
    pvalues = []
    
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1_data = dataframe[dataframe[group_col] == groups[i]][metric_col]
            group2_data = dataframe[dataframe[group_col] == groups[j]][metric_col]
            
            test_result = test_func(group1_data, group2_data, significance_level)
            pvalues.append(test_result['pvalue'])
            
            results.append({
                'group1': groups[i],
                'group2': groups[j],
                'statistic': test_result['statistic'],
                'pvalue': test_result['pvalue']
            })
    
    if correction_method:
        correction_func = getattr(corrections, f"{correction_method}_correction")
        corrected_pvalues = correction_func(pvalues, len(groups), significance_level)
        for i, result in enumerate(results):
            result['corrected_pvalue'] = corrected_pvalues[i]
            result['significant'] = corrected_pvalues[i] < significance_level
    
    return pd.DataFrame(results)