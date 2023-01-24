import numpy as np
import pandas as pd
 
try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
except HTTPError:
    df = pd.read_csv("/Users/vatsalsodha/ML_codes/machine-learning-book/ch02/iris.data", 
                     header=None,
                     encoding='utf-8')

df.tail()