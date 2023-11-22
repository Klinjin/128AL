import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from IPython import embed

def read_file(file_name,sheet_name):
    '''
    reading time data measured in nanoseconds between successive signals that triggered the readout electronics
    '''
    time=[]
    for i in sheet_name:
        data_excel = pd.read_excel(file_name, sheet_name=i)
        time.append(np.array(data_excel['data'].tolist()))
    return time

read_time=read_file('raw_data.xlsx',['trial1','trial2'])
filter_time=[]
for i in read_time:
    filter_time.append(i[i<4000-])

plt.hist(filter_time[0],bins=100)
plt.show()
def calc_lifetime(n, t):
    '''
    n: number of muons
    t: time interval
    '''
    return n/t