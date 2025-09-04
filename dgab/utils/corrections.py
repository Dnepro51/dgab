import pandas as pd
import numpy as np
from scipy import stats


def bonferroni_correction(p_values, n_groups, significance_level):
    """Bonferroni correction for multiple comparisons.
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
    """
    n_comparisons = n_groups * (n_groups - 1) // 2
    corrected_pvalues = [min(p * n_comparisons, 1.0) for p in p_values]
    return corrected_pvalues