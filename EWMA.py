import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
import numpy as np

class EWMA(object):
    def __init__(self, ret:np.ndarray, look_back_period:int, alpha:float, decay_factor:float):
        self.data = ret
        self.N = look_back_period
        self.alpha = alpha
        self.lmd = decay_factor
        
    def h(self, t):
        '''h(t) can be replaced by other expoential weighted functions'''
        return (self.lmd ** t * (1 - self.lmd)) / (1-self.lmd ** self.N)
    
    def prob(self):
        t_rank = np.arange(self.N, 0, -1)
        return self.h(t_rank)
    
    def ewma(self,x):
        x = pd.DataFrame(x,columns=['ret'])
        x['prob'] = self.prob()
        x = x.sort_values('ret')
        x['cumulative p'] = x['prob'].cumsum()
        if x.loc[x['cumulative p'] <= self.alpha].empty:
            return x.iloc[0, 0]
        else:
            return np.max(x.loc[x['cumulative p'] <= self.alpha, 'ret'])
    
    def fit(self):
        ret  = pd.DataFrame(self.data, columns=['ret'])
        ret['ewma'] = ret.rolling(window=self.N, center=False).apply(lambda x: self.ewma(x))
        self.var = ret['ewma']
        
    @property
    def res(self):
        return self.var
        
    
def clearData(df, col_name='vwretx'):
    df['caldt'] = pd.to_datetime(df['caldt'])
    df = df.set_index('caldt')
    return pd.DataFrame(df[col_name]).rename(columns={col_name:'ret'}) 
    
if __name__ == '__main__':
    data = pd.read_csv('sp500.csv')
    data = clearData(data)
    ewma = EWMA(ret=data, look_back_period=260, alpha=0.05, decay_factor=0.98)
    ewma.fit()
    print(ewma.res)
    plt.plot(ewma.res)
    plt.show()