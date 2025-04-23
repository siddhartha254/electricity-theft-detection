import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Define your split folders, ratio values, and desired colors
splits = {
    60: {'ratio': 0.6, 'color': 'red'},
    70: {'ratio': 0.7, 'color': 'blue'},
    80: {'ratio': 0.8, 'color': 'green'},
}

# Prepare storage for data
data = {}
for split, info in splits.items():
    ratio = info['ratio']
    folder = f"{split}_split"
    pattern = os.path.join(folder, f"ratio_{ratio}_run_0.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV found for split {split} at pattern {pattern}")

    df = pd.read_csv(files[0])
    data[split] = df

# Plot Train & Validation Accuracy vs Epoch
plt.figure(figsize=(10, 6))
for split, df in data.items():
    color = splits[split]['color']
    plt.plot(df['Epoch'], df['TrainAcc'], color=color, linestyle='-', label=f'{split}% Train Acc')
    plt.plot(df['Epoch'], df['ValAcc'], color=color, linestyle='--', label=f'{split}% Val Acc')

plt.title('Train and Validation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Train & Validation Loss vs Epoch
plt.figure(figsize=(10, 6))
for split, df in data.items():
    color = splits[split]['color']
    plt.plot(df['Epoch'], df['TrainLoss'], color=color, linestyle='-', label=f'{split}% Train Loss')
    plt.plot(df['Epoch'], df['ValLoss'], color=color, linestyle='--', label=f'{split}% Val Loss')

plt.title('Train and Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
