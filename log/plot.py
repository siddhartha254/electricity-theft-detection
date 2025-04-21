import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV file (make sure it's saved as 'metrics.csv' in the working directory)
df = pd.read_csv('summary_ratio_0.7_run_0.csv')

avg_train_acc = df['TrainAcc'].mean()
avg_val_acc   = df['ValAcc'].mean()
print(f'Average Train Accuracy: {avg_train_acc:.4f}')
print(f'Average Validation Accuracy: {avg_val_acc:.4f}')

# Plot 1: Train vs Validation Accuracy
plt.figure()
plt.plot(df['Epoch'], df['TrainAcc'], label='Train Accuracy')
plt.plot(df['Epoch'], df['ValAcc'], label='Validation Accuracy')

# Trendlines
z_train_acc = np.polyfit(df['Epoch'], df['TrainAcc'], 1)
p_train_acc = np.poly1d(z_train_acc)
plt.plot(df['Epoch'], p_train_acc(df['Epoch']), linestyle='--', label='Train Acc Trend')

z_val_acc = np.polyfit(df['Epoch'], df['ValAcc'], 1)
p_val_acc = np.poly1d(z_val_acc)
plt.plot(df['Epoch'], p_val_acc(df['Epoch']), linestyle='--', label='Val Acc Trend')

plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Train vs Validation Loss
plt.figure()
plt.plot(df['Epoch'], df['TrainLoss'], label='Train Loss')
#plt.plot(df['Epoch'], df['ValLoss'], label='Validation Loss')

# Trendlines
z_train_loss = np.polyfit(df['Epoch'], df['TrainLoss'], 1)
p_train_loss = np.poly1d(z_train_loss)
plt.plot(df['Epoch'], p_train_loss(df['Epoch']), linestyle='--', label='Train Loss Trend')

#z_val_loss = np.polyfit(df['Epoch'], df['ValLoss'], 1)
#p_val_loss = np.poly1d(z_val_loss)
#plt.plot(df['Epoch'], p_val_loss(df['Epoch']), linestyle='--', label='Val Loss Trend')

plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
