import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
# from .sr import *
#from .sim import *

def get_range_proba(predict, label, delay=7):
    '''
    根据延迟delay调整异常标签
    '''
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def extract_anomaly_nsigma(matrix_profile, label, mean, sigma, t, delay):
    '''
    根据 n sigma 原则提取异常
    '''
    n = len(matrix_profile)
    pred = np.zeros(n)
    anno_idx = matrix_profile >= (mean + t*sigma)
    pred[anno_idx] = 1
    y_pred = get_range_proba(pred, label, delay=delay)
    pred_dict = {
        'mp_pred': pred,
        'mp_pred_adjusted': y_pred,
        'groundtruth': label
    }
    fscore = f1_score(label, y_pred)
    pscore = precision_score(label, y_pred)
    rscore = recall_score(label, y_pred)

    return pred_dict, mean+3*sigma, fscore, pscore, rscore


def extract_anomaly_ratio(ratio, label, thresholds, delay):
    '''
    根据ratio提取异常
    '''
    n = len(ratio)
    pred = np.zeros(n)
    anno_idx = ratio >= thresholds
    pred[anno_idx] = 1
    y_pred = get_range_proba(pred, label, delay=delay)
    pred_dict = {
        'mp_pred': pred,
        'mp_pred_adjusted': y_pred,
        'groundtruth': label
    }
    fscore = f1_score(label, y_pred)
    pscore = precision_score(label, y_pred)
    rscore = recall_score(label, y_pred)

    return pred_dict, thresholds, fscore, pscore, rscore


