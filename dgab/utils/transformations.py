import pandas as pd
import numpy as np


def aggregate_to_individual_binary(dataframe, group_col, metric_config):
    """Convert aggregated binary data to individual 0/1 observations for t-test.

    Args:
        dataframe: DataFrame with aggregated binary data
        group_col: Column name containing group identifiers
        metric_config: Dict with 'trials_col_name' and 'successes_col_name'

    Returns:
        DataFrame with individual binary observations

    Example:
        Input:
        | group | users | conversions |
        |-------|-------|-------------|
        | A     | 100   | 20          |
        | B     | 150   | 45          |

        Output:
        | group | binary_outcome |
        |-------|----------------|
        | A     | 1              |
        | A     | 1              |
        | ...   | ...            |
        | A     | 0              |
        | B     | 1              |
        | ...   | ...            |

    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """
    individual_data = []

    for _, row in dataframe.iterrows():
        group = row[group_col]
        successes = int(row[metric_config['successes_col_name']])
        trials = int(row[metric_config['trials_col_name']])
        failures = trials - successes

        # Create individual observations: 1s for successes, 0s for failures
        group_data = [1] * successes + [0] * failures

        # Add to result with group labels
        for outcome in group_data:
            individual_data.append({group_col: group, 'binary_outcome': outcome})

    return pd.DataFrame(individual_data)