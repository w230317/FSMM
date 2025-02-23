# FSMM

This is "Fish School for Market Movement (FSMM)" algorithm
```python

# intall
# !pip install yfinance==0.2.54

# requirements
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, accuracy_score, matthews_corrcoef, f1_score
import yfinance as yf
import networkx as nx
import os
import random
import warnings
warnings.filterwarnings('ignore')

# import FSMM.py
from FSMM import *

# Example of use
symbols =  ((pd.read_csv('https://raw.githubusercontent.com/w230317/FSMM/refs/heads/main/IBOV_202502.csv', header=None, usecols=[0])))
symbols = symbols.sample(10, random_state=42)
symbol = symbols.sample(1, random_state=42)[0].astype(str)

# Step 1 - define hiperparameters
# Values for lag, λv , λd, step_ind, and step_vol
lag = 63
step_ind = 0.01
step_vol = 0.01
lambda_d = 1.4
lambda_v = 0.01

train_size = 6 # fsmm network correlation period in months
start_date = '01-01-2024'
end_date = '31-12-2024'


# Step 2 - load class
fsmm = FSMM()

# Step 3 - Generate database from Yahoo Finance. Pay attention to the date format dd-mm-yyyy
fsmm.gen_asset_data(symbols, lag, start_date, end_date, info=False)


# Step 4 - Load  data / FSMM network
returns = fsmm.df_from_csv('fsmm_df_daily_return.csv')
volumes = fsmm.df_from_csv('fsmm_df_daily_volume.csv')
network = fsmm.gen_fsmm_network(lambda_d, returns, start_date, train_size, lag)

# Backtest
metrics, stats = fsmm.fss_backtest(symbol, lag, start_date, end_date, lambda_v, step_ind, step_vol, returns, network, volumes)

# Forecasting
forecast = fsmm.fss_forecasting(symbol, lag, start_date, end_date, lambda_v, step_ind, step_vol, returns, network, volumes)


```
