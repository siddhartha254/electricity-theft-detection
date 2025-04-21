import pandas as pd

# 1. read the raw dump
df = pd.read_csv('data/data.csv')

# 2. extract labels
#    - CONS_NO is the customer ID (you usually won't train on that)
#    - FLAG is your theft/not‑theft target
labels = df[['CONS_NO', 'FLAG']].copy()
labels.to_csv('data/label.csv', index=False)

# 3. build your features matrix
#    drop the ID and FLAG columns
features = df.drop(columns=['CONS_NO', 'FLAG'])

# 4. (optional) ensure features.shape[1] is divisible by 7
n = features.shape[1]
drop = n % 7
if drop:
    # you can either drop the rightmost 'drop' columns:
    features = features.iloc[:, :-drop]
    # or pad with zeros if you’d rather keep the extra days:
    # pad = pd.DataFrame(0, index=features.index, columns=[f'PAD_{i}' for i in range(drop)])
    # features = pd.concat([features, pad], axis=1)

# 5. save your “after preprocessing” file
features.to_csv('data/after_preprocess_data.csv', index=False)

print("Saved:")
print("  • data/label.csv             ← (CONS_NO, FLAG)")
print("  • data/after_preprocess_data.csv ← all other columns ({} cols)".format(features.shape[1]))
