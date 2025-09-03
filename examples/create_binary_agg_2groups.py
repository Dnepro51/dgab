import pandas as pd
import numpy as np

np.random.seed(42)

# Generate binary aggregated data (conversion rates)
data = [
    {'group': 'A', 'users': 1200, 'conversions': 120},  # 10% conversion
    {'group': 'B', 'users': 1100, 'conversions': 143}   # 13% conversion
]

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('binary_agg_2groups.csv', index=False)
print(f"Created binary_agg_2groups.csv with {len(df)} rows")
print("Conversion rates by group:")
for _, row in df.iterrows():
    rate = (row['conversions'] / row['users']) * 100
    print(f"Group {row['group']}: {rate:.2f}% ({row['conversions']}/{row['users']})")