import pandas as pd
import numpy as np

np.random.seed(42)

# Generate binary aggregated data for multiple groups
data = [
    {'group': 'A', 'users': 1200, 'conversions': 120},  # 10.0% conversion
    {'group': 'B', 'users': 1000, 'conversions': 110},  # 11.0% conversion  
    {'group': 'C', 'users': 800,  'conversions': 96},   # 12.0% conversion
    {'group': 'D', 'users': 1100, 'conversions': 143},  # 13.0% conversion
    {'group': 'E', 'users': 900,  'conversions': 90}    # 10.0% conversion (same as A)
]

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('binary_agg_multiple.csv', index=False)
print(f"Created binary_agg_multiple.csv with {len(df)} rows")
print("Conversion rates by group:")
for _, row in df.iterrows():
    rate = (row['conversions'] / row['users']) * 100
    print(f"Group {row['group']}: {rate:.2f}% ({row['conversions']}/{row['users']})")