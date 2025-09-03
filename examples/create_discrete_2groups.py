import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

# Generate discrete data (launches per user)
group_a = np.random.poisson(1.0, n_samples)  # Control group
group_b = np.random.poisson(1.1, n_samples)  # Test group

# Create DataFrame
df = pd.DataFrame({
    'group': ['A'] * n_samples + ['B'] * n_samples,
    'launches': np.concatenate([group_a, group_b])
})

# Save to CSV
df.to_csv('discrete_2groups.csv', index=False)
print(f"Created discrete_2groups.csv with {len(df)} rows")
print(df.groupby('group')['launches'].agg(['count', 'mean', 'std']))