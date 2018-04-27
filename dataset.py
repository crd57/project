# -*- coding: utf-8 -*-

# @Time    : 18-2-2 下午4:43
# @Author  : Crd
# @Email   : crd57@126.com
# @File    : dataset.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib as mpl
import  matplotlib.pyplot as plt

def create_interval_dataset(dataset,look_back):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back])
    return [np.asarray(dataX), np.asarray(dataY)]


df = pd.read_csv("date.csv")
dataset_init = np.asarray(df)  # if only 1 column
[dataX, dataY] = create_interval_dataset(dataset_init,5)
mpl.rcParams['xtick.labelsize']=24
plt.figure('data')
plt.plot(dataY)
plt.show()
