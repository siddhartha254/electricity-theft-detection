from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class MyMetric(keras.callbacks.Callback):
    def __init__(self, train_ratio, num):
        super().__init__()
        self.X_val = None
        self.y_val = None
        self.train_ratio = train_ratio
        self.num = num
        self.epoch = 0
        
        # Metrics tracking
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_auc = []
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.map1 = []
        self.map2 = []

    def set_validation_data(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val

    def precision_at_k(self, r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        return np.mean(r) if r.size == k else 0.0

    def average_precision(self, r):
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k+1) for k in range(r.size) if r[k]]
        return np.mean(out) if out else 0.0

    def mean_average_precision(self, rs):
        return np.mean([self.average_precision(r) for r in rs]) if rs else 0.0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        # Extract logs
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('accuracy', logs.get('acc'))

        # Record metrics
        self.train_loss.append(loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_acc)

        # Validation predictions
        preds = self.model.predict(self.X_val, verbose=0)[:, 0]
        y = self.y_val[:, 0] if len(self.y_val.shape) > 1 else self.y_val
        mask = ~np.isnan(preds) & ~np.isnan(y)
        preds, y = preds[mask], y[mask]

        if len(preds) < 2 or len(np.unique(y)) < 2:
            # Skip metrics if invalid
            for lst in [self.val_auc, self.val_accuracy, self.val_precision,
                        self.val_recall, self.val_f1, self.map1, self.map2]:
                lst.append(np.nan)
        else:
            # Compute metrics
            auc = roc_auc_score(y, preds)
            y_pred = (preds >= 0.5).astype(int)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)

            self.val_auc.append(auc)
            self.val_accuracy.append(accuracy_score(y, y_pred))
            self.val_precision.append(prec)
            self.val_recall.append(rec)
            self.val_f1.append(f1)

            # MAP@100 & 200
            temp = pd.DataFrame({
                'label_0': y, 'label_1': 1-y,
                'preds_0': preds, 'preds_1': 1-preds
            })
            self.map1.append(self.mean_average_precision([
                temp.sort_values('preds_0', ascending=False).label_0[:100],
                temp.sort_values('preds_1', ascending=False).label_1[:100]
            ]))
            self.map2.append(self.mean_average_precision([
                temp.sort_values('preds_0', ascending=False).label_0[:200],
                temp.sort_values('preds_1', ascending=False).label_1[:200]
            ]))

        # Build CSV row with safe formatting
        def fmt(x): return f"{x:.4f}" if x is not None and not np.isnan(x) else ''
        row = [str(self.epoch), fmt(loss), fmt(val_loss), fmt(train_acc)]
        # add row for each val metric
        for arr in [self.val_auc, self.val_accuracy, self.val_precision,
                    self.val_recall, self.val_f1, self.map1, self.map2]:
            row.append(fmt(arr[-1] if arr else None))

        # Write header if first epoch
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        path = f'{log_dir}/ratio_{self.train_ratio:.1f}_run_{self.num}.csv'
        write_header = not os.path.exists(path)
        with open(path, 'a') as f:
            if write_header:
                header = ['Epoch','TrainLoss','ValLoss','TrainAcc','AUC','ValAcc',
                          'Precision','Recall','F1','MAP100','MAP200']
                f.write(','.join(header) + '\n')
            f.write(','.join(row) + '\n')

        # Print summary line
        print(f"Epoch {self.epoch}: loss={fmt(loss)}, val_loss={fmt(val_loss)}, "
              f"train_acc={fmt(train_acc)}, val_acc={fmt(self.val_accuracy[-1] if self.val_accuracy else None)}, "
              f"auc={fmt(self.val_auc[-1] if self.val_auc else None)}")

    def on_train_end(self, logs=None):
        epochs = list(range(1, self.epoch+1))

        # Loss plot
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.train_loss, 'b-o', label='Train Loss')
        plt.plot(epochs, self.val_loss, 'r-o', label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves (Ratio {self.train_ratio:.1f} Run {self.num})')
        plt.legend()
        plt.savefig(f'log/loss_ratio_{self.train_ratio:.1f}_run_{self.num}.png')
        plt.close()

        # Accuracy plot with percentage y-axis
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.train_accuracy, 'b-o', label='Train Accuracy')
        plt.plot(epochs, self.val_accuracy, 'r-o', label='Val Accuracy')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves (Ratio {self.train_ratio:.1f} Run {self.num})')
        plt.legend()
        plt.savefig(f'log/accuracy_ratio_{self.train_ratio:.1f}_run_{self.num}.png')
        plt.close()

        # Summary CSV of all metrics
        summary = pd.DataFrame({
            'Epoch': epochs,
            'TrainLoss': self.train_loss,
            'ValLoss': self.val_loss,
            'TrainAcc': self.train_accuracy,
            'AUC': self.val_auc,
            'ValAcc': self.val_accuracy,
            'Precision': self.val_precision,
            'Recall': self.val_recall,
            'F1': self.val_f1,
            'MAP100': self.map1,
            'MAP200': self.map2
        })
        summary.to_csv(f'log/summary_ratio_{self.train_ratio:.1f}_run_{self.num}.csv', index=False)
