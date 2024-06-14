from data_provider.data_loader import Dataset_KPI,Dataset_Yahoo,Dataset_NAB,Dataset_NASA,Dataset_Weather,Dataset_WSD
from torch.utils.data import DataLoader

data_dict = {
    'KPI':Dataset_KPI,
    'Yahoo':Dataset_Yahoo,
    'NAB' :Dataset_NAB,
    'MSL':Dataset_NASA,
    'Weather':Dataset_Weather,
    'WSD':Dataset_WSD
}
cur = 0


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'online':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.data == "KPI":
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_index,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_set=args.train_set,
            scale = args.scale
        )
    elif "Yahoo" in args.root_path:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_index,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            scale = args.scale
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            scale = args.scale
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
