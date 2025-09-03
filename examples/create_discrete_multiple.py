import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

# Generate discrete data for 5 groups (clicks per user)
group_a = np.random.poisson(2.0, n_samples)   # Control
group_b = np.random.poisson(2.1, n_samples)   # Small increase
group_c = np.random.poisson(2.2, n_samples)   # Medium increase  
group_d = np.random.poisson(2.3, n_samples)   # Large increase
group_e = np.random.poisson(2.0, n_samples)   # Same as control

# Create DataFrame
df = pd.DataFrame({
    'group': ['A'] * n_samples + ['B'] * n_samples + ['C'] * n_samples + 
              ['D'] * n_samples + ['E'] * n_samples,
    'clicks': np.concatenate([group_a, group_b, group_c, group_d, group_e])
})

# Save to CSV
df.to_csv('discrete_multiple.csv', index=False)
print(f"Created discrete_multiple.csv with {len(df)} rows")
print(df.groupby('group')['clicks'].agg(['count', 'mean', 'std']))