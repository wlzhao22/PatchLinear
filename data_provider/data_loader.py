import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils import *
import warnings
import uuid

warnings.filterwarnings('ignore')
class Dataset_KPI(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='value', scale=True, timeenc=0, freq='h', seasonal_patterns=None,train_set='true'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'val','test','online']
        type_map = {'train': 0, 'val': 1,'test':2,'online':3}
        self.set_type = type_map[flag]
        self.train_set = train_set
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_path = self.data_path +'.csv'
        test_path = self.data_path +'.csv'
        df_raw = pd.read_csv(os.path.join(self.root_path,'train',
                                          train_path))
        df_test = pd.read_csv(os.path.join(self.root_path,'test',
                                          test_path))
                                          
        # df_label = pd.read_csv(os.path.join(self.root_path,'label',
        #                                   test_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
       
        num_train = int(len(df_raw) * 0.7)
        if self.set_type==0 or self.set_type==1:
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else :
            border1s = [0]
            border2s = [len(df_test)]
            border1 = 0
            border2 = len(df_test)

        if self.set_type==0 or self.set_type==1:
            df_data = df_raw[[self.target]].values
            df_label = np.zeros_like(df_data)
        else:
            df_data = df_test[[self.target]].values
            df_label = df_test[['label']].values
        train_data = df_raw[[self.target]].values
        if self.scale:
            train_data = train_data[:num_train]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        # df_stamp = df_raw[['timestamp']][border1:border2]
        # df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['timestamp'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        data_stamp = np.array([np.arange(0,len(data))]).transpose()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]
        #print(index,seq_x.shape,s_begin,s_end,r_begin,r_end,len(self.data_x) - self.seq_len - self.pred_len + 1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Yahoo(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val','online']
        type_map = {'train': 0, 'val': 1, 'test': 2,'online':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        if "A3" in data_path or "A4" in data_path:
            self.root_path = root_path+data_path.split('_')[0]
            name = data_path.split('_')
            self.data_path = name[0]+"Benchmark-TS"+name[1]
        else:
            self.root_path = root_path+data_path.split('_')[0]
            self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['timestamp', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols + [self.target]]
        df_label = df_raw['label'].values
        # print(cols)
        num_train = int(len(df_raw) * 0.35)
        num_vali = int(len(df_raw) * 0.15)
        num_test = len(df_raw) - num_train - num_vali
        border1s = [0, num_train + 1,len(df_raw) - num_test]
        border2s = [num_train, len(df_raw) - num_test - 1 , len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]
    
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        #return len(self.data_x) - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_NAB(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='value', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val','online']
        type_map = {'train': 0, 'val': 1, 'test': 2,'online':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path+'.csv'))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('timestamp')
        df_label = df_raw['label'].values
        df_raw = df_raw[['timestamp'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.35)
        num_val = int(len(df_raw)*0.5)
        num_test = len(df_raw) - num_train
        border1s = [0,num_train-self.seq_len,num_val-self.seq_len]
        border2s = [num_train, num_val,len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end #- self.label_len
        #r_end = r_begin + self.label_len + self.pred_len
        r_end = r_begin+self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_NASA(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='value', scale=True, timeenc=0, freq='h', seasonal_patterns=None,train_set='true'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'val','test','online']
        type_map = {'train': 0, 'val': 1,'test':2,'online':2}
        self.set_type = type_map[flag]
        self.train_set = train_set
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_path = 'train/'+self.data_path +'.npy'
        test_path =  'test/'+self.data_path+'.npy'
        df_raw = np.load(os.path.join(self.root_path,
                                          train_path))[:,:1].reshape(-1,1)
        df_test = np.load(os.path.join(self.root_path,
                                          test_path))[:,:1].reshape(-1,1)
                                          
        df_label = pd.read_csv(os.path.join(self.root_path,'test_labels',
                                          self.data_path+'.csv'),header=None)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.8)
        num_vali = len(df_raw) - num_train 
        if self.set_type==0 or self.set_type==1:
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1s = [0]
            border2s = [len(df_test)]
            border1 = 0
            border2 = len(df_test)

        if self.set_type==0 or self.set_type==1:
            df_data = df_raw
            df_label = np.zeros_like(df_data)
        elif self.set_type==2:
            df_data = df_test
            df_label = df_label.values
        else:
            df_data = df_raw[-(self.seq_len-1):]
            df_data = np.concatenate((df_data,df_test),axis=0)
            temp = np.zeros((self.seq_len-1,1))
            df_label = np.concatenate((temp,df_label.values),axis=0)
            border2 = border2+self.seq_len-1
        train_data = df_raw
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = np.array([np.arange(0,len(self.data_x))]).transpose()
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Weather(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='value', scale=True, timeenc=1, freq='h', seasonal_patterns=None,train_set='true'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'val','test','online']
        type_map = {'train': 0, 'val': 1,'test':2,'online':3}
        self.set_type = type_map[flag]
        self.train_set = train_set
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_path = 'new_temperature.csv'
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          train_path))     
        # df_label = pd.read_csv(os.path.join(self.root_path,'label',
        #                                   test_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
       
        num_train = int(len(df_raw) * 0.5)
        num_test = len(df_raw) - num_train
        border1s = [0,0,num_test]
        border2s = [num_train, num_train , len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[[self.data_path]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # df_stamp = df_raw[['datetime']][border1:border2]
        # df_stamp['datetime'] = pd.to_datetime(df_stamp.datetime)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.datetime.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.datetime.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.datetime.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['datetime'], axis=1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        df_stamp = np.arange(0,len(df_raw))
        data_stamp = df_stamp
        df_label = np.zeros(len(df_raw))
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]
        #print(index,seq_x.shape,s_begin,s_end,r_begin,r_end,len(self.data_x) - self.seq_len - self.pred_len + 1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_WSD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val','online']
        type_map = {'train': 0, 'val': 1, 'test': 2,'online':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['timestamp', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('timestamp')
        df_raw = df_raw[['timestamp'] + cols + [self.target]]
        df_label = df_raw['label'].values
        # print(cols)
        num_train = int(len(df_raw) * 0.35)
        num_vali = int(len(df_raw) * 0.15)
        num_test = len(df_raw) - num_train - num_vali
        border1s = [0, num_train + 1,len(df_raw) - num_test]
        border2s = [num_train, len(df_raw) - num_test - 1 , len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = df_label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_label = self.data_label[r_end-self.pred_len:r_end]
    
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label
    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1
        #return len(self.data_x) - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)