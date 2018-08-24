# coding: utf-8

import load_data

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class VAR(object):
    def __init__(self,data):
        self.data = data
        self.time = data.shape[0]
        self.k = data.shape[1]
        self.p = 0
        self.beta = None
        self.X = None

    def var(self, p = 2):
        self.p = p
        if self.p * self.k >= self.time - self.p:
            print('Need more sample size.')
        else:
            self.X = np.zeros([self.time-self.p,self.p * self.k])
            p_list = list(range(1, self.p + 1))
            index = [str(d) + ',t-' + str(p_list[-lag]) for d in range(self.k) for lag in p_list]
            columns = range(self.k)
            df_beta = pd.DataFrame(index=index,columns=columns)
            for i in range(self.k):
                for t in range(self.p, self.time):
                    # data[t,i] = data[t-1:t-p,:]
                    self.X[t-self.p,:] = self.data[t-self.p:t,:].reshape(1,self.p * self.k,order='F')
                beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.data[self.p:,i]
                df_beta.iloc[:,i] = beta
            self.beta = np.array(df_beta)
            return df_beta

    def granger_causality(self):
        SSR = (self.data[self.p:,:] - self.X @ self.beta).T @ (self.data[self.p:,:] - self.X @ self.beta)
        SSR = np.diag(SSR)
        F_test = np.zeros([self.k,self.k])
        for i,j in itertools.product(range(self.k),range(self.k)):
            F_bunshi = (SSR[i] - SSR[j])/2
            F_bunbo = SSR[j]/(self.time - self.k*self.p -1)
            F = F_bunshi/F_bunbo
            F_test[i,j] = F
        return F_test

    def impuls_function(slef):
        return 0


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
    model = VAR(np.array(D))
    result = model.var(p=3)
    print(result)
    print('F',model.granger_causality())

