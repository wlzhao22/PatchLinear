import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
import json
import os

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def mul(value, label):
    vl = np.zeros((len(value)))
    for i in range(len(value)):
        if label[i]:
            vl[i] = value[i] * label[i]
        else:
            vl[i] = float('inf')
    return vl
def draw_prediction(y_pred,y_true,savepath):
    plt.rcParams['figure.figsize'] = (200, 4)
    plt.cla()
    plt.clf()
    plt.close()

    ax1 = plt.subplot(1, 1, 1)
    plt.plot(y_true, label='truth')
    plt.plot(y_pred, label='predict', alpha=.7)
    plt.ylim([min(min(y_true), min(y_pred)) - 0.2, max(max(y_true), max(y_pred)) + 0.2])
    plt.legend(loc=2, fontsize='xx-large')
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    plt.savefig(savepath)
def draw_plot(mse_score,true,test_label,pred,thresh,preds,savepath):
    plt.subplots(4,1,figsize=(20, 20))
    plt.subplot(4,1,1)
    plt.plot(range(len(true)),true,label='truth')
    index = np.where(test_label==1)
    plt.scatter(index,true[index],c='r')
    plt.legend()
    plt.subplot(4,1,2)
    plt.plot(range(len(true)),true,label='pred')
    index = np.where(pred==1)
    plt.scatter(index,true[index],c='r')
    plt.legend()
    plt.subplot(4,1,3)
    plt.plot(range(len(true)),true,label='truth')
    plt.plot(range(len(true)),preds,label='pred',alpha=.7)
    plt.legend()
    plt.subplot(4,1,4)
    plt.plot(range(len(true)),mse_score,label='mae')
    plt.plot(range(len(true)),thresh,label='thresh')
    plt.legend()
    plt.savefig(savepath)
def draw_plot_mp(mse_score,true,test_label,pred,preds,ratio,thresh,savepath):
    plt.clf()
    plt.subplots(5,1,figsize=(20, 10))
    plt.subplot(5,1,1)
    plt.plot(range(len(true)),true,label='truth')
    index = np.where(test_label==1)
    plt.scatter(index,true[index],c='r')
    plt.legend()
    plt.subplot(5,1,2)
    plt.plot(range(len(true)),true,label='pred')
    index = np.where(pred==1)
    plt.scatter(index,true[index],c='r')
    index = np.where(pred==2)
    plt.scatter(index,true[index],c='g')
    plt.legend()
    plt.subplot(5,1,3)
    plt.plot(range(len(true)),true,label='truth')
    plt.plot(range(len(true)),preds,label='pred',alpha=.7)
    plt.legend()
    plt.subplot(5,1,4)
    plt.plot(range(len(true)),mse_score,label='mae')
    plt.plot(range(len(true)),thresh,label='thresh',alpha=.7)
    plt.legend()
    plt.subplot(5,1,5)
    plt.plot(range(len(ratio)),ratio,label='ratio')
    plt.scatter(index,ratio[index],c='r')
    plt.legend()
    plt.savefig(savepath)
def get_range_proba(predict, label, delay=7):
    '''
    根据延迟delay调整异常标签。
    '''

    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:
        if 1 in predict[pos:min(pos + delay + 1, sp)]:
            new_predict[pos:sp] = 1
        else:
            new_predict[pos:sp] = 0
    return new_predict

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
def compute_mean_std(series, window_size, cache_window):
    '''
    计算窗口的滑窗均值和标准差
    '''
    n = len(series)
    if cache_window<n:
        flag = True
    else:
        flag = False
    series_cum = np.cumsum(series)
    window_mean = np.zeros(n, dtype=float)

    c_window_mean = np.zeros(n, dtype=float)

    window_mean[window_size:] = series_cum[window_size:] - series_cum[:-window_size]
    window_mean /= window_size
    for i in range(1, window_size):
        window_mean[i] = series_cum[i] / (i+1)
    #window_mean[1:]=window_mean[:-1]
    window_mean[0] = series[0]
    if flag:
        c_window_mean[cache_window:] = series_cum[cache_window:] - series_cum[:-cache_window]
        c_window_mean /= cache_window
        for i in range(1, cache_window):
            c_window_mean[i] = series_cum[i] / (i+1)
        #window_mean[1:]=window_mean[:-1]
        c_window_mean[0] = series[0]

    series_square_cum = np.cumsum(series**2)
    window_std = np.zeros(n, dtype=float)
    c_window_std = np.zeros(n, dtype=float)
    window_std[window_size:] = (series_square_cum[window_size:] - series_square_cum[:-window_size])/window_size - window_mean[window_size:]**2
    for i in range(1, window_size):
        window_std[i] = series_square_cum[i]/(i+1) - window_mean[i]**2
    #window_std[1:] = window_std[:-1]
    window_std[0] = 0
    window_std[window_std<0] = 0
    window_std = np.sqrt(window_std)
    if flag:
        c_window_std[cache_window:] = (series_square_cum[cache_window:] - series_square_cum[:-cache_window])/cache_window - c_window_mean[cache_window:]**2
        for i in range(1, cache_window):
            c_window_std[i] = series_square_cum[i]/(i+1) - c_window_mean[i]**2
        #window_std[1:] = window_std[:-1]
        c_window_std[0] = 0
        c_window_std[c_window_std<0] = 0
        c_window_std = np.sqrt(c_window_std)

    return window_mean, window_std, c_window_mean, c_window_std

def compute_mean_std_mul(series, window_size):
    '''
    计算窗口的滑窗均值和标准差,多时序
    '''
    n = len(series)
    series_cum = np.cumsum(series,axis = 0)
    window_mean = np.zeros_like(series)
    window_mean[window_size:] = series_cum[window_size:] - series_cum[:-window_size]
    window_mean /= window_size
    for i in range(1, window_size):
        window_mean[i] = series_cum[i] / (i+1)
    window_mean[0] = series[0]

    series_square_cum = np.cumsum(series**2,axis=0)
    window_std = np.zeros_like(series)
    window_std[window_size:] = (series_square_cum[window_size:] - series_square_cum[:-window_size])/window_size - window_mean[window_size:]**2
    for i in range(1, window_size):
        window_std[i] = series_square_cum[i]/(i+1) - window_mean[i]**2
    window_std[0] = 0
    window_std[window_std<0] = 0
    window_std = np.sqrt(window_std)

    return window_mean, window_std
if __name__=='__main__': 
    p = np.random.rand(8000,5)
    p1 = p[:,1]
    mean1,std1 = compute_mean_std(p1,21)
    mean,std = compute_mean_std_mul(p,21)
    print(mean1 == mean[:,1])
    print(std1 == std[:,1])
    print(mean.shape,std.shape)

    

    



