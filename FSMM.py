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


class FSMM:

  def __init__(self):
    pass

  def df_from_csv(self, file_csv):

    if self.gen_asset_data_error():
      return None

    df = pd.read_csv(file_csv)
    retornos_mensal = df
    df.index = pd.to_datetime(df['Date'])
    df = df.drop('Date', axis=1)
    return df


  def gen_asset_data_error(self):
    file_daily_close = 'fsmm_df_daily_close.csv'
    if not os.path.exists(file_daily_close):
      print('ERROR: you need to generate the asset data to continue...')
      return True
    else:
      return False

  def df_from_csv(self, file_csv):

    if self.gen_asset_data_error():
      return None


    df = pd.read_csv(file_csv)
    retornos_mensal = df
    df.index = pd.to_datetime(df['Date'])
    df = df.drop('Date', axis=1)
    return df

  def gen_asset_data(self, symbols, lag, start_date, end_date, info=True):
      # generate asset data
      file_monthly_return = 'fsmm_df_monthly_return.csv'
      file_daily_return = 'fsmm_df_daily_return.csv'
      file_daily_close = 'fsmm_df_daily_close.csv'
      file_daily_volume = 'fsmm_df_daily_volume.csv'
      file_raw_data = 'fsmm_df_raw_data.csv'

      # remove older files if exists
      if os.path.exists(file_monthly_return):
          os.remove(file_monthly_return)
      if os.path.exists(file_daily_close):
          os.remove(file_daily_close)
      if os.path.exists(file_daily_volume):
          os.remove(file_daily_volume)
      if os.path.exists(file_raw_data):
          os.remove(file_raw_data)
      if os.path.exists(file_daily_return):
          os.remove(file_daily_return)

      # convert list/dataframe of assets to array
      if type(symbols)==pd.core.frame.DataFrame:
        symbols = np.array(symbols[0])
      else:
        symbols = np.array(symbols)

      cols=[]

      # get data from yahoo fynance

      start = pd.to_datetime(start_date) - pd.DateOffset(days=lag+365)
      # start = datetime.strptime(start_date, '%d-%m-%Y')
      end = datetime.strptime(end_date, '%d-%m-%Y')

      df_daily_close  = pd.date_range(start, end , freq='D')
      df_daily_close = [d.strftime('%Y-%m-%d') for d in df_daily_close]
      df_daily_close = pd.DataFrame(df_daily_close)
      df_daily_close.columns=['Date']
      df_daily_close.index = df_daily_close['Date']

      df_daily_volume  = pd.date_range(start, end , freq='D')
      df_daily_volume = [d.strftime('%Y-%m-%d') for d in df_daily_volume]
      df_daily_volume = pd.DataFrame(df_daily_volume)
      df_daily_volume.columns=['Date']
      df_daily_volume.index = df_daily_volume['Date']

      df_daily_return  = pd.date_range(start, end , freq='D')
      df_daily_return = [d.strftime('%Y-%m-%d') for d in df_daily_return]
      df_daily_return = pd.DataFrame(df_daily_return)
      df_daily_return.columns=['Date']
      df_daily_return.index = df_daily_return['Date']

      df_monthly_return = pd.date_range(start, end , freq='1ME')
      df_monthly_return = [d.strftime('%Y-%m-01') for d in df_monthly_return]
      df_monthly_return = pd.DataFrame(df_monthly_return)
      df_monthly_return.columns=['Date']
      df_monthly_return.index = df_monthly_return['Date']


      df_raw_data = pd.DataFrame()


      for s in symbols:

        try:
          if info:
            print(start, end)
            print('Stock '+ s)

          ticker = yf.Ticker(s)
          df = ticker.history(start=start, end=end)
          df_raw = df.reset_index()
          df_raw['ticker'] = s
          df_raw_data = pd.concat([df_raw_data, df_raw])

          p = df['Close'] # price
          p = p.fillna(0.0)
          p = p[p != 0]

          v = df['Volume'] # volume
          v = v.fillna(0.0)

          if (len(p)) > 0:

            monthly_return = pd.DataFrame(p.resample('ME').ffill().pct_change().fillna(0.0))
            monthly_return.index = monthly_return.index.strftime("%Y-%m-01")
            df_monthly_return[s] = pd.DataFrame(monthly_return)
            df_monthly_return = df_monthly_return.fillna(0.0)


            daily_return =  (np.log(p) - np.log(p.shift(1))).fillna(0.0)  # log return
            daily_return.index = [d.strftime('%Y-%m-%d') for d in p.index]
            df_daily_return[s] = pd.DataFrame(daily_return)
            df_daily_return = df_daily_return.fillna(0.0)


            p.index = [d.strftime('%Y-%m-%d') for d in p.index]
            v.index = [d.strftime('%Y-%m-%d') for d in v.index]

            df_daily_close[s] = pd.DataFrame(p)
            df_daily_close = df_daily_close.fillna(0.0)

            df_daily_volume[s] = pd.DataFrame(v)
            df_daily_volume = df_daily_volume.fillna(0.0)

            cols.append(s)
        except:
          pass
          if info:
            print('ERROR ' + s)

      df_monthly_return = df_monthly_return.drop('Date', axis=1)
      df_daily_close = df_daily_close.drop('Date', axis=1)
      df_daily_volume = df_daily_volume.drop('Date', axis=1)
      df_daily_return = df_daily_return.drop('Date', axis=1)

      df_daily_close = df_daily_close.loc[(df_daily_close.sum(axis=1) != 0), (df_daily_close.sum(axis=0) != 0)]
      df_daily_volume = df_daily_volume.loc[(df_daily_volume.sum(axis=1) != 0), (df_daily_volume.sum(axis=0) != 0)]
      df_daily_return = df_daily_return.loc[(df_daily_return.sum(axis=1) != 0), (df_daily_return.sum(axis=0) != 0)]

      df_monthly_return.to_csv(file_monthly_return)
      df_daily_close.to_csv(file_daily_close)
      df_daily_volume.to_csv(file_daily_volume)
      df_daily_return.to_csv(file_daily_return)

      df_raw_data.to_csv(file_raw_data)


      return 'Finished ' + str(len(cols)) +' of '+ str(len(symbols))

  def gen_df_lag(self, symbol, returns, lag):

    if self.gen_asset_data_error():
      return None

    df = pd.DataFrame(returns[symbol]).reset_index()
    df.columns = ['t', 'return']
    cols = []
    df['target'] = df['return'].shift(-1)  # next return
    features = ['return']
    for f in features:
      for lag in range(0, lag + 1):
          col = f'{f}_lag_{lag}'
          df[col] = df[f].shift(lag)
          cols.append(col)
    df = df.tail(-lag).fillna(0).reset_index()
    df.drop('index', axis=1, inplace=True)
    return df


  def gen_df_lag_volume(self, symbol, volumes, lag):
    df = pd.DataFrame(volumes[symbol]).reset_index()
    df.columns = ['t', 'volume']
    cols = []
    df['target'] = df['volume'].shift(-1)  # next
    features = ['volume']
    for f in features:
      for lag in range(0, lag + 1):
          col = f'{f}_lag_{lag}'
          df[col] = df[f].shift(lag)
          cols.append(col)
    df = df.tail(-lag).fillna(0).reset_index()
    df.drop('index', axis=1, inplace=True)
    return df


  def gen_fsmm_network(self, lambda_d, returns, start_date, train_size, lag):

    symbols = returns.columns
    df_corr = pd.concat([self.gen_df_lag(s, returns, lag)[(self.gen_df_lag(s, returns, lag)['t'] > pd.to_datetime(start_date)-pd.DateOffset(months=(train_size))) & (self.gen_df_lag(s, returns, lag)['t'] < start_date)].iloc[:, 3:lag+4].mean(axis=1) for s in symbols], axis=1)
    df_corr.columns = symbols

    M = df_corr.fillna(0).corr() # correlation matrix M
    np.fill_diagonal(M.values, 0)

    # Build a network, G, through the matrix M
    G = nx.Graph(M)
    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    #fish network
    fnet = pd.DataFrame(G.edges(data='weight'))
    # graph dataframe includes one-way row, e.g. v -> u
    fnet.columns = ['from','to','weight']
    fnet = fnet[fnet.weight != 0.0]
    xndf = fnet.copy()
    # reverse order and concatenate, to create the loop, u -> v
    ndfi = xndf[['to','from','weight']]
    ndfi.columns = ['from','to','weight']

    xnet = pd.concat([xndf, ndfi])

    # distance d_ij
    # correlation transformation -1 is mapped to 2 and 1 to 0
    xnet['d_ij'] = [np.sqrt(2*(1-(w))) for w in xnet['weight']]
    xnet = xnet[(xnet['d_ij'] <= lambda_d)]
    return xnet

  # Basic functions
  def transition(self, updws):
      # receives a vector of 0.1, indicating a fall and rise in price
      # returns transition vector with frequency values ​​for transitions [0-1,1-1,0-0,1-0]

      y = np.roll(updws,1) # next value
      x = np.delete(updws,0) # delete first value
      y = np.delete(y,0)
      v = y - 2*x   # maps comb. 0 and 1 in 4 values, (0 1)=-2; (1 1)=-1; (0 0)=0; (1 0) = 1
      unique, counts = np.unique(v, return_counts=True)
      f = np.zeros(4) # creates vector with 0; It may be that the sequence does not have all combinations
      unique = unique+2 # sum 2 for single values ​​to become vector index -2=0, -1=1, 0=2, 1=3
      f[unique] = counts
      return f

  def moving_transition(self, a, n=10):
      # sliding window of size n, calculates transition vector based on the window
      l = a.shape
      stdmov=[]
      for i in range(l[0]-n+1):
          stdmov.append(self.transition(a[i:n+i]))

      return np.array(stdmov)

  # FSMM movements / swimming reviewed

  def fss_individual_behavior_markov(self, symbol, lag, start_date, step_ind, returns):

    df_lag = self.gen_df_lag(symbol, returns, lag)
    df_lag = df_lag[df_lag['t']>=start_date]
    tam = df_lag.shape[0]+(lag-1)

    df_lag = self.gen_df_lag(symbol, returns, lag)
    df_lag['avg'] = df_lag.iloc[0:, 3:lag+4].mean(axis=1)
    df_lag['std'] = df_lag.iloc[0:, 3:lag+4].std(axis=1)
    df_lag = df_lag.tail(tam)

    rets = np.array(df_lag['return'].values)
    labels = np.array(rets > np.roll(rets,-1)).astype(int)
    matTrans = self.moving_transition(labels, lag)  # transition matrix
    (lin, col) = matTrans.shape
    prob = np.random.rand(lin)

    est1 = [1 if (prob[x] < matTrans[x,1]/(matTrans[x,1]+matTrans[x,3])) else 0 for x in range(lin) ] # cols 1 e 3 representam ação no estado 1
    est0 = [1 if (prob[x] < matTrans[x,0]/(matTrans[x,0]+matTrans[x,2])) else 0 for x in range(lin) ] # 0 e 2 no estado 0
    # consider the state of the action

    movm = [est0[x] if labels[x] == 0 else est1[x] for x in range(lin)] # rise/fall prediction
    df_lag = df_lag[df_lag['t']>=start_date]

    df_lag['avg'] = df_lag.iloc[0:, 3:lag+4].mean(axis=1)
    df_lag['std'] = df_lag.iloc[0:, 3:lag+4].std(axis=1) 
    df_lag['rand'] = random.gauss(df_lag['avg'], df_lag['std'])
    df_lag['movm'] = np.array(movm)
    retCut = np.array(df_lag['avg'])
    rand = np.array(df_lag['rand'])
    df_lag['x_it_next'] = [((retCut[x]-step_ind)*rand[x]) if movm[x] == 0 else (retCut[x]+step_ind)*rand[x] for x in range(lin)]
    df_lag['x_it_next'] = np.roll(df_lag['x_it_next'], 1) # np.roll -> prever o proximo movimento

    df_ind = np.array(df_lag['x_it_next']).reshape(-1)
    df_ind = round(pd.DataFrame(df_ind).fillna(0), 8)
    df_ind.columns = ['mov_ind']
    df_ind = pd.DataFrame(df_ind['mov_ind'])

    return df_ind


  def fss_volitive_behavior(self, symbol, lag, start_date, step_vol, returns, network):

    symbols =  network['from'].unique() # all symbols
    symbols = np.delete(symbols, np.where(symbols == symbol.item())) # x_jt

    df_ret = returns[returns.index>=start_date]

    r_it = pd.DataFrame(df_ret[symbol])
    r_it_prev = pd.DataFrame(df_ret[symbol].shift(1).fillna(0))
    r_it_calc = r_it - r_it_prev

    r_it_calc.reset_index(inplace=True)
    r_it_calc.drop('Date', axis=1, inplace=True)
    r_it_calc_t = r_it_calc.T

    r_jt = df_ret[symbols]
    r_jt_prev = df_ret[symbols].shift(1).fillna(0)
    r_jt_calc = r_jt - r_jt_prev

    r_jt_calc.reset_index(inplace=True)
    r_jt_calc.drop('Date', axis=1, inplace=True)
    r_jt_max = pd.DataFrame(r_jt_calc.max(axis=0))
    r_jt_calc_t = r_jt_calc.T

    df_weights = pd.DataFrame(pd.DataFrame(r_it_calc_t.to_numpy()/r_jt_max.to_numpy()).T.sum(axis=1))
    df_weights.columns = ['cur_weight'] # current weight

    x_jt = pd.concat([self.gen_df_lag(s, returns, lag)[self.gen_df_lag(s, returns, lag)['t'] >= start_date].iloc[:, 3:lag+4].mean(axis=1) for s in symbols], axis=1)
    x_jt.columns = symbols

    x_it = self.gen_df_lag(symbol, returns, lag)[self.gen_df_lag(symbol, returns, lag)['t'] >= start_date].iloc[:, 3:lag+4].mean(axis=1)
    x_it = pd.DataFrame(x_it, columns=[symbol])

    df_barycenter = pd.DataFrame(x_jt.to_numpy() * df_weights.to_numpy()).sum(axis=1) / x_jt.to_numpy().sum()

    df_weights['prev_weight'] = df_weights['cur_weight'].shift(1).fillna(0) # previous weight
    df_weights['increased'] = np.array(df_weights['cur_weight'] > df_weights['prev_weight']).astype(int)

    df_weights['x_it'] = np.array(x_it.values).reshape(-1)
    df_weights['rand'] = [np.random.rand() for _ in range(df_weights.shape[0])]
    df_weights['barycenter'] = df_barycenter.values
    df_weights['x_it_next'] = np.roll(df_weights['x_it'] - step_vol * df_weights['rand'] * (df_weights['x_it'] - df_weights['barycenter']), 1)
    if df_weights.loc[df_weights['increased']==1].empty:
        df_weights['x_it_next'] = np.roll(df_weights['x_it'] + step_vol * df_weights['rand'] * (df_weights['x_it'] - df_weights['barycenter']), 1)

    df_vol = np.array(df_weights['x_it_next']).reshape(-1)
    df_vol = round(pd.DataFrame(df_vol).fillna(0), 8)
    df_vol.columns = ['mov_vol']
    df_vol = pd.DataFrame(df_vol['mov_vol'])

    return df_vol


  def fss_instinctive_behavior(self, symbol, lag, start_date, returns, network):

    neighbors =  network[network['from']==symbol.item()]['to']

    x_jt = pd.concat([self.gen_df_lag(v, returns, lag)[self.gen_df_lag(v, returns, lag)['t'] >= start_date].iloc[:, 3:lag+4].mean(axis=1) for v in neighbors], axis=1)
    x_jt.columns = neighbors

    x_jt_prev = x_jt.shift(1).fillna(0) # previous
    x_jt_calc = x_jt - x_jt_prev

    df_ret = returns[returns.index>=start_date]
    r_jt = df_ret[neighbors]
    r_jt_prev = df_ret[neighbors].shift(1).fillna(0)
    r_jt_calc = r_jt - r_jt_prev # previous

    r_jt_calc.reset_index(inplace=True)
    r_jt_calc.drop('Date', axis=1, inplace=True)

    x_it_next = np.roll(pd.DataFrame((x_jt_calc.to_numpy()*r_jt_calc.to_numpy())).sum(axis=1) / r_jt_calc.sum(axis=1), 1)

    df_lag = self.gen_df_lag(symbol, returns, lag)
    df_lag = df_lag[df_lag['t']>=start_date]
    df_lag['x_it_next'] = np.array(x_it_next).reshape(-1)
    df_lag['x_it_next'] = np.roll(df_lag['x_it_next'], 1)

    df_col = np.array(df_lag['x_it_next']).reshape(-1)
    df_col = round(pd.DataFrame(df_col).fillna(0), 8)
    df_col.columns = ['mov_col']
    df_col = pd.DataFrame(df_col['mov_col'])

    return df_col


  def fss_choice_movement(self, symbol, lag, start_date, lambda_v, fss_ind, fss_vol, fss_col, volumes):

    dfv_lag = self.gen_df_lag_volume(symbol, volumes, lag)
    dfv_lag['avg'] = dfv_lag.iloc[0:, 3:lag+4].mean(axis=1)
    dfv_lag['avg_prev'] = dfv_lag['avg'].shift(1)
    dfv_lag = dfv_lag[dfv_lag['t']>=start_date]
    dfv_lag['movement_volume'] = 1-(dfv_lag['avg_prev'] / dfv_lag['avg'])
    df2 = pd.concat([dfv_lag.reset_index(), fss_ind, fss_vol, fss_col], axis=1)

    number_iterations = df2.shape[0]

    df_movements = pd.DataFrame()

    for t in range(number_iterations):
      movement_volume = df2['movement_volume'][t]

      if abs(movement_volume) > lambda_v and movement_volume <= 0:
          x_it = df2['mov_col'][t]
          movement ='instinctive'
      elif abs(movement_volume) > lambda_v and movement_volume >= 0:
          x_it = df2['mov_vol'][t]
          movement ='volitive'
      else:
          x_it = df2['mov_ind'][t]
          movement ='individual'

      x_it = pd.DataFrame([x_it])
      x_it['movement'] = [movement]
      x_it['t'] = t
      df_movements = pd.concat([df_movements, x_it])

    df_movements.columns = ['r_it_next', 'movement', 't']

    return df_movements

  def metrics_monthly(self, df_results, fss_mov):

      df_results['year_month'] = pd.to_datetime(df_results['t']).dt.to_period('M')
      df_results['m_it_cur'] = pd.DataFrame(df_results['target'] > (df_results['return'])).astype(int)
      df_results['m_it_next']  = pd.DataFrame(df_results['target'] > (fss_mov['r_it_next'].values.reshape(-1))).astype(int)
      df_final = pd.DataFrame()
      for m in df_results['year_month'].unique():
        r_pred = np.array(df_results[df_results.year_month==m][:df_results.shape[0]-1]['m_it_next']) # delete last row
        r_true = np.array(df_results[df_results.year_month==m][:df_results.shape[0]-1]['m_it_cur'])

        # Calculating the metrics
        precision = precision_score(r_true, r_pred)
        recall = recall_score(r_true, r_pred)
        accuracy = accuracy_score(r_true, r_pred)
        mcc = matthews_corrcoef(r_true, r_pred)
        f1 = f1_score(r_true, r_pred, average='binary')
        # Results
        d = ({
            'year_month' : [m],
            'precision' : [precision],
            'recall' : [recall],
            'acc' : [accuracy],
            'mcc' : [mcc],
            'f1' : [f1]
            })



        dfx = pd.DataFrame(d).reset_index(drop=True)
        df_final = pd.concat([df_final, dfx], axis=0)

      return df_final    


  def fss_backtest(self, symbol, lag, start_date, end_date, lambda_v, step_ind, step_vol, returns, network, volumes):

    fss_ind = self.fss_individual_behavior_markov(symbol, lag, start_date, step_ind, returns)
    fss_vol = self.fss_volitive_behavior(symbol, lag, start_date, step_vol, returns, network)
    fss_col = self.fss_instinctive_behavior(symbol, lag, start_date, returns, network)

    fss_mov = self.fss_choice_movement(symbol, lag, start_date, lambda_v, fss_ind, fss_vol, fss_col, volumes)    

    df_results = self.gen_df_lag(symbol, returns, lag)[self.gen_df_lag(symbol, returns, lag)['t'] >= start_date]
    metrics = self.metrics_monthly(df_results, fss_mov)

    df_numeric = metrics.iloc[:, 1:]

    mean_values = df_numeric.mean()
    std_values = df_numeric.std()

    mean_values.index = [f'{col}_avg' for col in mean_values.index]
    std_values.index = [f'{col}_std' for col in std_values.index]

    stats = pd.concat([mean_values.reset_index(), std_values.reset_index()], axis=1)
    stats.columns = np.array([[symbol]*4]).reshape(-1)
    return metrics, stats

  def fss_forecasting(self, symbol, lag, start_date, end_date, lambda_v, step_ind, step_vol, returns, network, volumes):

    fss_ind = self.fss_individual_behavior_markov(symbol, lag, start_date, step_ind, returns)
    fss_vol = self.fss_volitive_behavior(symbol, lag, start_date, step_vol, returns, network)
    fss_col = self.fss_instinctive_behavior(symbol, lag, start_date, returns, network)

    fss_mov = self.fss_choice_movement(symbol, lag, start_date, lambda_v, fss_ind, fss_vol, fss_col, volumes)    

    df_results = self.gen_df_lag(symbol, returns, lag)[self.gen_df_lag(symbol, returns, lag)['t'] >= start_date]

    forecast = pd.concat([df_results.reset_index(), fss_mov.reset_index()], axis=1).tail(1)
    forecast['m_it_cur'] = pd.DataFrame(forecast['target'] > (forecast['return'])).astype(int)
    forecast['m_it_next']  = pd.DataFrame(forecast['target'] > (forecast['r_it_next'].values.reshape(-1))).astype(int)

    dt = pd.to_datetime(str(forecast['t'].values[0][0])).strftime('%Y-%m-%d')
    msg = symbol + ' - forecasting for the next business day after ' + dt + ': '+ str([-1 if m==0 else 1 for m in forecast['m_it_next']][0])    
    
    print(msg.values[0])
    
    return msg




