# coding: utf-8

import load_data

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def var(data, p = 2):
    time = data.shape[0]
    k = data.shape[1]
    if p*k < time-p:
        X = np.zeros([time-p,p*k])
        p_list = list(range(1,p+1))
        index = [str(k) + ',t-' + str(p_list[-lag]) for k in range(k) for lag in p_list]
        columns = range(k)
        df_beta = pd.DataFrame(index=index,columns=columns)
        for i in range(k):
            for t in range(p,data.shape[0]):
                # data[t,i] = data[t-1:t-p,:]
                X[t-p,:] = data[t-p:t,:].reshape(1,p*k,order='F')
            beta = np.linalg.inv(X.T @ X) @ X.T @ data[p:,i]
            df_beta.iloc[:,i] = beta
        return df_beta
    else:
        print('Need more sample size.')

if __name__ == '__main__':
    df = load_data.get_data()
    D = df.iloc[:,:5]
    #                 AAPL UW Equity  AXP UN Equity  BA UN Equity  CAT UN Equity  CSCO UW Equity
    # Dates
    # 2015-01-30        0.061424      -0.132739      0.118403      -0.126297       -0.052130
    # 2015-02-27        0.096449       0.011154      0.037697       0.036639        0.119287
    # 2015-03-31       -0.031372      -0.042530     -0.005104      -0.034620       -0.067265
    # 2015-04-30        0.005786      -0.008577     -0.044909       0.085593        0.047411
    # 2015-05-29        0.040991       0.029309     -0.019674      -0.017956        0.016649
    print(var(np.array(D),p = 5 ))

