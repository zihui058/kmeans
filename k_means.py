# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:43:39 2019

@author: IAlpaca
"""

import os
import numpy as np
import pandas as pd
import random

def get_distance(x_series,target_index):
    return np.abs(x_series-x_series[target_index])

def get_total_dis(class_series,distan_df):
    for i in range(distan_df.shape[1]):
        distan_df[i] = distan_df[i] * (class_series == i)
    return distan_df.sum().sum()

def MyKMeans(data,k):
    n = len(data)
    step = 0
    min_distance = 1000
    while step < 1000:
        pick_point = sorted(random.sample(range(n),k))
        distan_df = pd.DataFrame()
        for i in range(k):
            distan_df[i] = get_distance(data,pick_point[i])
        class_series = pd.Series(np.array(distan_df).argmin(axis=1))
        dist = get_total_dis(class_series,distan_df)
        if dist < min_distance:
#            print(step,dist)            
            step = 0
            min_distance = dist
            mark_class = class_series
            mark_centre = pick_point
        step += 1
    return mark_class,mark_centre

def graph_weight(data):
    graph_w = pd.DataFrame(index=data.index,columns=data.index)
    for i in data.index:
        graph_w[i] = np.abs(data - data.iloc[i])
    return graph_w

def graph_threshold(graph_w):
    pass_threshold = graph_w*0
    for i in graph_w.columns:
        pass_threshold[i] = graph_w[i] < graph_w[i].sum()/99
    return pass_threshold.replace(False,np.nan) * graph_w

def graph_attachment(graph_new_w):
    return graph_new_w.sum()/graph_new_w.count()

# 加载数据
os.chdir('file_path')    #在引号内输入文件所在路径
data = pd.read_csv('ret_cluster.csv')
data.columns = ['stock','rpct','industry']
graph_w = graph_weight(data['rpct'])
graph_new_w = graph_threshold(graph_w)
data['attachment'] = graph_attachment(graph_new_w)
data['attach_clss'],centre_point1 = MyKMeans(data['attachment'],4)
data['rpct_class'],centre_point2 = MyKMeans(data['rpct'],4)
data.to_csv('cluster_result.csv',encoding='gbk')

