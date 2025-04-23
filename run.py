from datetime import datetime
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from function import *
from keras_metric import *
from wide_cnn import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/after_preprocess_data_scaled.csv')
    label = pd.read_csv('data/label.csv').rename(columns=str.lower)
    
    # Experiment parameters
    train_ratios = [0.6]  # Add more ratios as needed
    num_runs = 1

    for ratio in train_ratios:
        print(f'\n{"="*40}\nTraining with ratio {ratio:.1f}\n{"="*40}')
        
        # Split data
        X_train_wide, X_test_wide, y_train, y_test = train_test_split(
            data.values, label.flag.values, 
            test_size=1-ratio,
             
            random_state=2017
        )
        
        # Reshape for deep inputs
        X_train_deep = X_train_wide.reshape(-1, 1, X_train_wide.shape[1]//7, 7).transpose(0,2,3,1)
        X_test_deep = X_test_wide.reshape(-1, 1, X_test_wide.shape[1]//7, 7).transpose(0,2,3,1)
        
        # Preprocess
        X_train_pre = self_define_cnn_kernel_process(X_train_deep)
        X_test_pre = self_define_cnn_kernel_process(X_test_deep)
        
        # Model parameters
        weeks, days, channel = X_train_deep.shape[1:4]
        wide_len = X_train_wide.shape[1]

        # Multiple runs
        for run in range(num_runs):
            print(f'\n{"-"*30}\nRun {run+1}/{num_runs}\n{"-"*30}')
            
            # Initialize model
            model = Wide_CNN(weeks, days, channel, wide_len)
            
            # Setup metrics
            metric_cb = MyMetric(train_ratio=ratio, num=run)
            metric_cb.set_validation_data([X_test_wide, X_test_pre], y_test)
            
            # Train model
            history = model.fit(
                [X_train_wide, X_train_pre], y_train,
                validation_data=([X_test_wide, X_test_pre], y_test),
                epochs=200,
                batch_size=64,
                verbose=1,
                callbacks=[metric_cb]
            )