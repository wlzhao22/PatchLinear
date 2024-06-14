import numpy as np
import warnings
# from util import *

warnings.filterwarnings('ignore')
def get_best_index(series,mp,c_size,qt,means,vars,index):
    if len(series)>=mp:
        means = np.append(means,np.mean(series[-mp:]))
        vars = np.append(vars,np.var(series[-mp:]))
    if len(series)<mp:
        pass
    elif len(series)<=c_size:
        if len(series)==mp:
            cur_qt = np.dot(series[:mp],series[-mp:])
            qt = np.append(qt,cur_qt)
            dist = np.ones(1)*np.inf
            dist[0] = 0
        else:
            qt = qt - series[:len(qt)]*series[len(qt)-1]  + series[mp:mp+len(qt)]*series[-1]
            qt = np.append(qt,np.dot(series[:mp],series[-mp:]))
            qt = np.roll(qt,1)
            dist = np.real(np.sqrt(mp*(vars+vars[-1])-2*(qt-mp*means*means[-1])).astype(complex))
            # dist = np.real(np.sqrt(2*mp*(1-(qt-mp*means[-1]*means)/(mp*vars[-1]*vars))).astype(complex))
        dist[-int(mp/4):] = np.inf
        index = np.append(index,np.argmin(dist))
    else:
        cache = series[-c_size:]
        n = len(series)-c_size
        var = vars[-1]
        mean = means[-1]
        qt = qt - cache[-mp-1]*series[-c_size-1:-c_size-1+len(qt)] + cache[-1]*cache[mp-c_size-1:]
        # distance_profile = np.real(np.sqrt(2 * m * (1 - (qt_first-m*window_mean_b[0]*window_mean_a)/(m*window_std_b[0]*window_std_a))).astype(complex))
        # distance_profile = np.real(np.sqrt(m*(np.square(window_std_b[0])+np.square(window_std_a))-2*(qt-m*window_mean_b[0]*window_mean_a)).astype(complex))
        # dist = np.real(np.sqrt(2 * mp * ( 1 - ( qt - mp * mean * means[-c_size-1 + mp:])/(mp * var * vars[-c_size - 1 + mp:]))).astype(complex))
        dist = np.real(np.sqrt(mp*(vars[-c_size-1+mp:]+var)-2*(qt-mp*means[-c_size-1+mp:]*mean)).astype(complex))
        dist[-int(mp/4):] = np.inf
        index = np.append(index,n+np.argmin(dist))
    return qt,means,vars,index


if __name__=='__main__':
    import pandas as pd
    value = pd.read_csv('~/dataset/yahoo/real/real_46.csv')['value'].values
    m = 48
    sigma = []
    for i in range(len(value)-m+1):
        sigma.append(np.std(value[i:i+m]))
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(sigma)
    plt.subplot(212)
    plt.plot(value)
    plt.savefig('./test.png')