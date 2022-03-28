# DO NOT MODIFY

import numpy as np
import pandas as pd

np.random.seed(17)

df = pd.DataFrame()
df['X0'] = np.random.uniform(0,1,100)
df['y'] = np.random.normal(0,1,100)
df.to_csv('data/dataset1.csv')

df = pd.DataFrame()
df['X0'] = np.random.uniform(0,1,100)
df['X1'] = np.random.uniform(2,3,100)
df['X2'] = np.random.uniform(-2,2,100)
df['y'] = np.random.normal(0,10,100)
df.to_csv('data/dataset2.csv')

df = pd.DataFrame()
df['X0'] = np.random.uniform(0,1,100)
df['X1'] = np.random.uniform(2,3,100)
df['y'] = df['X0']+2*df['X1']
df.to_csv('data/dataset3.csv')

print('Supported datasets are: data/dataset1.csv, data/dataset2.csv, data/dataset3.csv')