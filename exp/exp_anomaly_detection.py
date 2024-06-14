from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, NLinear, PatchTST,PatchLinear,iTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop,adjust_predicts,get_range_proba,compute_mean_std, draw_plot, draw_plot_mp, draw_prediction
from utils.metrics import metric
# from utils.stamp import get_stompi_cache,stomp,stompi_cache
from utils.index import get_best_index
from sklearn.metrics import precision_recall_fscore_support,f1_score,precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from utils.sr import SR


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
import json

import warnings
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings('ignore')

### point adjustment
def adjust_by_mp(pred,true, mp_idx, ratio, label, series_window, thresholds, delay, dis_ratio):
    n = len(true)
    m = series_window
    for i in range(len(mp_idx)):
        if pred[mp_idx[i]+m-1] == 1 :
            ##MAD
            cur_series = true[i:i+m]
            median_v = np.median(cur_series)
            abs_dev = np.absolute(cur_series-median_v)
            med_abs_dev = np.median(abs_dev)
            score = 0.674 * abs_dev / med_abs_dev
            if score[-1] > 5:
                pred[i+m-1] = 1
        elif ratio[i] >= thresholds:
            pred[i+m-1] = 1
    y_pred = get_range_proba(pred, label, delay=delay)

    return y_pred
##distance significance
def get_ratio(mp_idx,true,m):  
    ratio = np.zeros(len(mp_idx))
    for j in range(len(mp_idx)):
        idx = mp_idx[j]
        if m>30:
            rate_num = 30
        else:
            rate_num = m
        cur_sub = true[j:j+m][-rate_num:]
        idx_sub = true[idx:idx+m][-rate_num:]
        dis = (cur_sub - np.mean(cur_sub) - idx_sub + np.mean(idx_sub))**2
        ratio[j] =(dis[-1])/(np.sum(dis)+1e-8)
    return ratio
class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': PatchLinear,
            'PatchTST': PatchTST,
            'PatchLinear':PatchLinear,
            'iTransformer':iTransformer
        }
        model = model_dict[self.args.model].Model(self.args).double()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset_dict={
            'KPI':['0efb375b-b902-3661-ab23-9a0bb799f4e3','1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0','4d2af31a-9916-3d9f-8a8e-8a268a48c095','05f10d3a-239c-3bef-9bdc-a2feeb0037aa','6a757df4-95e5-3357-8406-165e2bd49360','6d1114ae-be04-3c46-b5aa-be1a003a57cd',
                    '6efa3a07-4544-34a0-b921-a155bd1a05e8','9c639a46-34c8-39bc-aaf0-9144b37adfc8','42d6616d-c9c5-370a-a8ba-17ead74f3114','55f8b8b8-b659-38df-b3df-e4a5a8a54bc9','301c70d8-1630-35ac-8f96-bc1b6f4359ea','431a8542-c468-3988-a508-3afd06a218da',
                    '847e8ecc-f8d2-3a93-9107-f367a0aab37d','7103fa0f-cac4-314f-addc-866190247439','8723f0fb-eaef-32e6-b372-6034c9c04b80','43115f2a-baeb-3b01-96f7-4ea14188343c','54350a12-7a9d-3ca8-b81f-f886b9d156fd','57051487-3a40-3828-9084-a12f7f23ee38',
                    'a07ac296-de40-3a7c-8df3-91f642cc14d0','a8c06b47-cc41-3738-9110-12df0ee4c721','ab216663-dcc2-3a24-b1ee-2c3e550e06c9','adb2fde9-8589-3f5b-a410-5fe14386c7af','ba5f3328-9f3f-3ff5-a683-84437d16d554','c69a50cf-ee03-3bd7-831e-407d36c7ee91',
                    'c02607e8-7399-3dde-9d28-8a8da5e5d251','da10a69f-d836-3baa-ad40-3e548ecf1fbd','e0747cad-8dc8-38a9-a9ab-855b61f5551d','f0932edd-6400-3e63-9559-0a9860a1baa9','ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa'],
            'Yahoo':[67,100,100,100],
            'MSL':['C-1', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-7',  'T-4',  'T-8', 'T-9', 'T-12', 'T-13'],
            'NAB':['Twitter_volume_AAPL','Twitter_volume_AMZN','Twitter_volume_CRM','Twitter_volume_CVS','Twitter_volume_FB','Twitter_volume_GOOG','Twitter_volume_IBM','Twitter_volume_KO','Twitter_volume_PFE','Twitter_volume_UPS']
        }
        data_index = dataset_dict[self.args.data]
        sets = []
        loaders = []
        if 'Yahoo' in self.args.data:
            dataset = ['real_','synthetic_','A3_','A4_']
            for i in range(len(data_index)): 
                for j in range(1,data_index[i]+1):
                    self.args.data_index = dataset[i]+str(j)+".csv"
                    data_set, data_loader = data_provider(self.args, flag)
                    sets.append(data_set)
                    loaders.append(data_loader)
        elif 'WSD' in self.args.data:
            for i in range(data_index[0]):
                self.args.data_path = str(i)+".csv"
                data_set, data_loader = data_provider(self.args, flag)
                sets.append(data_set)
                loaders.append(data_loader)
        else:
            for item in data_index:
                if 'KPI' in self.args.data:
                    self.args.data_index = item
                else:
                    self.args.data_path = item
                data_set, data_loader = data_provider(self.args, flag)
                sets.append(data_set)
                loaders.append(data_loader)
        return sets, loaders
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=1e-3)
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,_) in enumerate(vali_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)
                 # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Transformer' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)    
                else:
                    if 'Transformer' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs,batch_y)
                loss = loss.detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()
        train_steps = len(train_loader)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        start_time = time.time()
        for j in range(len(train_loader)):
            path = os.path.join(self.args.checkpoints, setting,str(j))
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            self.model = self._build_model().to(self.device)
            model_optim = self._select_optimizer()
            criterion = self._select_criterion()
            if not os.path.exists(path):
                os.makedirs(path)
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []
                max_bound = 0
                self.model.train()
                epoch_time = time.time()
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,label) in enumerate(train_loader[j]):
                     # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double().to(self.device)
                    iter_count += 1
                    batch_x = batch_x.double().to(self.device)
                    batch_y = batch_y.double().to(self.device)
                    batch_x_mark = batch_x_mark.double().to(self.device)
                    batch_y_mark = batch_y_mark.double().to(self.device)
                    # # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Transformer' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x)
                            f_dim = -1 if self.args.features == 'S' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_x = batch_x[:, -self.args.pred_len:, f_dim:].to(self.device)
                            
                            loss = criterion(ouputs,batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Transformer' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                        # print(outputs.shape,batch_y.shape)
                        f_dim = -1 if self.args.features == 'S' else 0     
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                       
                        
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                    if (i + 1) % 100 == 0:
                        #print("\titers: {0}, epoch: {1} | loss: {2:.10f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        #print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    model_optim.zero_grad()
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                if 'Weather' in self.args.data:
                    vali_loss = train_loss
                else:
                    vali_loss = self.vali(vali_data[j], vali_loader[j], criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.10f} Vali Loss: {3:.10f}".format(
                    epoch + 1, j, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        end_time = time.time()
        print("training stage cost {0}s.".format(end_time-start_time))

        return 0

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        _,train_loader = self._get_data(flag='train')
        preds_arr = []
        true_arr = []
        attens_energy_arr = []
        test_labels_arr = []
        ratio_arr = []
        index_arr = []
        mses = []
        maes = []
        score_means = []
        score_stds = []
        rses = []
        model_optim = self._select_optimizer()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        self.model.use_multiprocessing = False
        self.model.eval()
        dim = 0
        t1 = 0
        start = time.time()
        for j in range(len(test_loader)):
            preds = []
            trues = []
            attens_energy = []
            test_labels = []
            ratio = []
            dist = []
            series = np.array([],dtype=np.float64)
            qt = np.array([],dtype=np.float64)
            means = np.array([],dtype=np.float64)
            vars = np.array([],dtype=np.float64)
            index = np.array([],dtype=np.int16)
            ratios = np.array([],dtype=np.float64)
            first_flag = True
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting,str(j),'checkpoint.pth')))
            with torch.no_grad():
                scores = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_label) in enumerate(train_loader[j]):
                    batch_x = batch_x.double().to(self.device)
                     # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double().to(self.device)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Transformer' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x)
                    else:
                        if 'Transformer' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                    f_dim=-1
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    score = torch.abs( outputs - batch_y ) 
                    outputs = outputs.detach().cpu().numpy()
                    scores.append(score.detach().cpu().numpy())
                mean = np.mean(scores)
                std = np.std(scores)
            score_means.append(mean)
            score_stds.append(std)
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_label) in enumerate(test_loader[j]):
                    batch_x = batch_x.double().to(self.device)
                     # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double().to(self.device)
                    if dim==0:
                        dim = batch_x.shape[-1]
                    batch_y = batch_y.double().to(self.device)
                    batch_x_mark = batch_x_mark.double().to(self.device)
                    batch_y_mark = batch_y_mark.double().to(self.device)
                    # decoder input
                    # encoder - decoder

                    start_time = time.time()
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Transformer' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x)
                    else:
                        if 'Transformer' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x)
                    end_time = time.time()
                    t1 = t1 + end_time - start_time
                    f_dim = -1 if self.args.features == 'S' else 0    
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    test_labels.append(y_label)
                    score = torch.abs( outputs - batch_y ) 
                    batch_y = batch_y.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                    score = score.detach().cpu().numpy()
                    attens_energy.append(score)
                    preds.append(pred)
                    trues.append(true)
                    if 'ds' in self.args.score_calc: 
                        series = np.append(series,true.reshape(-1))
                        if batch_x.shape[0] > 1: 
                            for i in range(batch_x.shape[0]):
                                qt,means,vars,index=get_best_index(series[:-(batch_x.shape[0]-i)],self.args.dp_window,self.args.cache_window,qt,means,vars,index)
                        else:
                            qt,means,vars,index=get_best_index(series,self.args.dp_window,self.args.cache_window,qt,means,vars,index)
            if 'ds' in self.args.score_calc:  
                index_arr.append(index)
            attens_energy = np.round(np.concatenate(attens_energy,axis=0),decimals=6)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)
            preds_arr.append(preds)
            true_arr.append(trues)
            test_labels_arr.append(test_labels)
            attens_energy_arr.append(attens_energy)
            temp = trues
            mae, mse, rmse, mape, mspe, rse= metric(preds, trues)
            trues = temp
            mses.append(mse)
            maes.append(mae)
            rses.append(rse)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        end = time.time()
        print("average mse:{0}, mae:{1}, rse:{2}, cost {3}s".format(np.average(mses),np.average(maes),np.average(rses),end-start_time))
        if self.args.plot == 'y':
            for i in range(len(preds_arr)):
                y_true = true_arr[i][:,0,0]
                y_pred = preds_arr[i][:,0,0]
                if not os.path.exists('./savefig/'+self.args.data+'/'):
                     os.makedirs('./savefig/'+self.args.data+'/')
                savepath = './savefig/'+self.args.data+'/'+self.args.model+'_prediction_'+str(i)+'.pdf'
                draw_prediction(y_pred,y_true,savepath)
        if self.args.anomaly_detection == 'y':
            self.anomaly_detection(attens_energy_arr,test_labels_arr,preds_arr,true_arr,score_means,score_stds,index_arr)
        end = time.time()
        print('forecasting cost '+str(t1)+' seconds')
        print('test stage cost '+str(end-start)+' seconds')
    def online_test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='online')
        _,train_loader = self._get_data(flag='train')
        preds_arr = []
        true_arr = []
        attens_energy_arr = []
        test_labels_arr = []
        index_arr = []
        mp_arr = []
        mses = []
        maes = []
        score_means = []
        score_stds = []
        rses = []
        model_optim = self._select_optimizer()
        start = time.time()
        criterion = nn.MSELoss(reduce=False)
        for j in range(len(test_loader)):
            preds = []
            trues = np.array([],dtype=np.float64)
            attens_energy = []
            test_labels = []
            qt = np.array([],dtype=np.float64)
            means = np.array([],dtype=np.float64)
            vars = np.array([],dtype=np.float64)
            index = np.array([],dtype=np.int16)
            ratios = np.array([],dtype=np.float64)
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting,str(j),'checkpoint.pth')))
            with torch.no_grad():
                scores = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_label) in enumerate(train_loader[j]):
                    batch_x = batch_x.double().to(self.device)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                    f_dim=-1
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    score = torch.abs( outputs - batch_y ) 
                    outputs = outputs.detach().cpu().numpy()
                    scores.append(score.detach().cpu().numpy())
                mean = np.mean(scores)
                std = np.std(scores)
            score_means.append(mean)
            score_stds.append(std)
            series = np.array([])
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,y_label) in enumerate(test_loader[j]):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'S' else 0    
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                model_optim.zero_grad()
                loss = criterion(outputs,batch_y)
                loss.backward()
                model_optim.step()
                test_labels.append(y_label)
                score = torch.abs( outputs - batch_y ) 
                batch_y = batch_y.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                preds.append(pred)
                trues = np.append(trues,true)
                if 'ds' in self.args.score_calc: 
                    qt,means,vars,index=get_best_index(true,self.args.dp_window,self.args.cache_window,qt,means,vars,index)
            if 'ds' in self.args.score_calc:  
                index_arr.append(index)
                print("in")
            attens_energy = np.round(np.concatenate(attens_energy,axis=0),decimals=6)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)
            preds_arr.append(preds)
            true_arr.append(trues)
            # mp_arr.append(matrix_profile)
            test_labels_arr.append(test_labels)
            attens_energy_arr.append(attens_energy)
            temp = trues
            mae, mse, rmse, mape, mspe, rse= metric(preds, trues)
            trues = temp
            mses.append(mse)
            maes.append(mae)
            rses.append(rse)
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print("average mse:{0}, mae:{1}, rse:{2}".format(np.average(mses),np.average(maes),np.average(rses)))
        if self.args.plot == 'y':
            for i in range(len(preds_arr)):
                y_true = true_arr[i][:,0,0]
                y_pred = preds_arr[i][:,0,0]
                if not os.path.exists('./savefig/'+self.args.data+'/'):
                     os.makedirs('./savefig/'+self.args.data+'/')
                savepath = './savefig/'+self.args.data+'/'+self.args.model+'_prediction_'+str(i)+'.pdf'
                draw_prediction(y_pred,y_true,savepath)
        if self.args.anomaly_detection == 'y':
            self.anomaly_detection(attens_energy_arr,test_labels_arr,preds_arr,true_arr,score_means,score_stds,index_arr)
        end = time.time()
        print('test stage cost '+str(end-start)+' seconds')
    def anomaly_detection(self,attens_energy_arr,test_labels_arr,preds_arr,true_arr,score_means,score_stds,index_arr=None):
        if self.args.score_calc == 'mae':
            best_f1 = 0
            best_p = 0
            best_r = 0
            best_thresh = 3.0
            best_y_pred = []
            best_thresh_arr = []
            thresh_mean = []
            thresh_std = []
            ## select a best result from different k
            for var in np.arange(3.0,7.1,0.1):
                pred_all = np.array([])
                label_all = np.array([])
                left_mp_adjusted_arr = []
                y_pred_arr = []
                tmp_diff_arr = []
                thresh_arr = []
                for i in range(len(attens_energy_arr)):
                    mse_score = attens_energy_arr[i].reshape(-1)
                    test_label = test_labels_arr[i].reshape(-1)
                    preds = preds_arr[i].reshape(-1)
                    true = true_arr[i].reshape(-1)
                    mean = score_means[i]
                    std = score_stds[i]
                    ##moving threshold
                    cur_mean,cur_std,cache_mean,cache_std = compute_mean_std(mse_score,self.args.dp_window,self.args.cache_window)
                    thresh = cur_mean+var*cur_std+1e-8
                    init_thresh = np.zeros(thresh.shape)
                    if thresh.shape[0] > self.args.cache_window:
                        init_thresh[:self.args.cache_window] = mean+var*std
                        init_thresh[self.args.cache_window:] = cache_mean[self.args.cache_window:]+var*cache_std[self.args.cache_window:]
                    else:
                        init_thresh[:] = score_means[i]+var*score_stds[i]
                    anno_idx = thresh<init_thresh
                    thresh[anno_idx] = init_thresh[anno_idx]
                    anno_idx = mse_score >= thresh 
                    thresh_arr.append(thresh)
                    pred = np.zeros_like(mse_score)
                    pred[anno_idx] = 1
                    y_pred = get_range_proba(pred, test_label, delay=self.args.delay)
                    y_pred_arr.append(y_pred)
                    fscore = f1_score(test_label, y_pred)
                    pscore = precision_score(test_label, y_pred)
                    rscore = recall_score(test_label, y_pred)
                    pred_all = np.append(pred_all,y_pred)
                    label_all = np.append(label_all,test_label)
                    print("Precision: {}, Recall: {}, F-score: {}".format(pscore,rscore,fscore))
                fscore = f1_score(label_all, pred_all)
                pscore = precision_score(label_all, pred_all)
                rscore = recall_score(label_all, pred_all)
                print("avg Precision: {}, Recall: {}, F-score: {} in {}".format(pscore,rscore,fscore,var))
                if fscore>best_f1:
                    best_f1 = fscore
                    best_p = pscore
                    best_r = rscore
                    best_thresh = var
                    best_thresh_arr = thresh_arr
                    best_y_pred = y_pred_arr
            print("best Precision: {}, Recall: {}, F-score: {} in {}".format(best_p,best_r,best_f1,best_thresh))
            if self.args.plot == 'y':
                for i in range(len(attens_energy_arr)):
                    if not os.path.exists('./savefig/'+self.args.data+'/'):
                        os.makedirs('./savefig/'+self.args.data+'/')
                    savepath = './savefig/'+self.args.data+'/'+str(i)+'.png'
                    mse_score = attens_energy_arr[i].reshape(-1)
                    true = true_arr[i].reshape(-1)
                    test_label = test_labels_arr[i]
                    pred = best_y_pred[i]
                    thresh = best_thresh_arr[i]
                    preds = preds_arr[i].reshape(-1)
                    draw_plot(mse_score,true,test_label,pred,thresh,preds,savepath)
        ##using distance significance
        elif self.args.score_calc == 'mae_ds':
            best_thresh = 0
            best_f1 = 0
            best_p = 0
            best_r = 0
            best_y_pred = []
            best_tmp_diff = []
            best_ratio_arr = []
            best_thresh_arr = []
            ratio_arr = []
            for i in range(len(attens_energy_arr)):
                true = true_arr[i].reshape(-1)
                index = index_arr[i]
                ratio = get_ratio(index,true,self.args.dp_window)    
                ratio_arr.append(ratio)  
            for var in np.arange(3.0,7.1,0.1):
                pred_all = np.array([])
                label_all = np.array([])
                y_pred_arr = []
                thresh_arr = []
                for i in range(len(attens_energy_arr)):
                    mse_score = attens_energy_arr[i]
                    mse_score = mse_score.reshape(-1)
                    test_label = test_labels_arr[i].reshape(-1)
                    preds = preds_arr[i].reshape(-1)
                    true = true_arr[i].reshape(-1)
                    mean = score_means[i]
                    std = score_stds[i]
                    ratio = ratio_arr[i]
                    index = index_arr[i]
                    cur_mean,cur_std,cache_mean,cache_std = compute_mean_std(mse_score,self.args.dp_window,self.args.cache_window)
                    thresh = cur_mean+var*cur_std+1e-8
                    init_thresh = np.zeros(thresh.shape)
                    if thresh.shape[0] > self.args.cache_window:
                        init_thresh[:self.args.cache_window] = mean+var*std
                        init_thresh[self.args.cache_window:] = cache_mean[self.args.cache_window:]+var*cache_std[self.args.cache_window:]
                    else:
                        init_thresh[:] = mean+var*std
                    anno_idx = thresh<init_thresh
                    thresh[anno_idx] = init_thresh[anno_idx]
                    anno_idx = mse_score >= thresh 
                    thresh_arr.append(thresh)
                    pred = np.zeros(len(true))
                    pred[anno_idx] = 1
                    y_pred = adjust_by_mp(pred,true, index, ratio, test_label, self.args.dp_window,self.args.ds_thresh , self.args.delay,self.args.dis_ratio)
                    y_pred_arr.append(y_pred)
                    y_pred[y_pred>1]=1
                    fscore = f1_score(test_label, y_pred)
                    pscore = precision_score(test_label, y_pred)
                    rscore = recall_score(test_label, y_pred)
                    pred_all = np.append(pred_all,y_pred)
                    label_all = np.append(label_all,test_label)
                    print("Precision: {}, Recall: {}, F-score: {}".format(pscore,rscore,fscore))
                fscore = f1_score(label_all, pred_all)
                pscore = precision_score(label_all, pred_all)
                rscore = recall_score(label_all, pred_all)
                print("avg Precision: {}, Recall: {}, F-score: {} in {}".format(pscore,rscore,fscore,var))
                if fscore>best_f1:
                    best_f1 = fscore
                    best_p = pscore
                    best_r = rscore
                    best_thresh = var
                    best_ratio_arr = ratio_arr
                    best_y_pred = y_pred_arr
                    best_thresh_arr = thresh_arr
            print("best Precision: {}, Recall: {}, F-score: {} in {}".format(best_p,best_r,best_f1,best_thresh))


        
                    
