import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

class EWMA(object):
    def __init__(self, ret:np.array, look_back_period:int, alpha:float, decay_factor:float):
        self.data = ret
        self.N = look_back_period
        self.alpha = alpha
        self.lmd = decay_factor
        
    def h(self):
        '''h(t) can be replaced by other expoential weighted functions'''
        return (self.lmd ** self.t * (1 - self.lmd)) / (1-self.lmd ** self.N)
    
    def __prob(self):
        self.t = 
        