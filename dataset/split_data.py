'''
此脚本文件主要是将20多G的文件分割为小文件，最后处理成一个船一个文件，方便之后数据处理

'''
import pandas as pd
from tqdm import tqdm
import os


train_columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude', 'vesselMMSI', 'speed',
                 'direction', 'vesselNextport','vesselNextportETA', 'vesselStatus', 'vesselDatasource',
                 'TRANSPORT_TRACE']


def split_train_data():
    '''
    将23G的文件1bw的小文件
    :return:
    '''
    train_data_path = r'D:\baiph\BDC2020\data\train0711/train0711.csv'
    df = pd.read_csv(train_data_path, header=None, chunksize=1000000)
    count = 1
    for chunk in df:
        chunk.columns = train_columns
        chunk.to_csv(r'D:\baiph\BDC2020\data\train\split-data/train%.3d.csv' % (count), index=None)
        count += 1


def to_data_MMSI():
    '''
    将每个分割的文件按照MMSI分组存储
    :return:
    '''
    data_path = r'D:\baiph\BDC2020\data\train\split-data/train%.3d.csv'
    output_path = r'D:\baiph\BDC2020\data\train\preprocess'
    for i in tqdm(range(1, 147)):
        temp_path = data_path % i
        # 得到 train_000 作为目录
        name = temp_path.split('.')[0].split('/')[-1]
        if not os.path.exists(os.path.join(output_path, name)):
            os.makedirs(os.path.join(output_path, name))
        # 读取数据
        datas = pd.read_csv(data_path % i, names=train_columns)
        # 按照船艘vesselMMSI分组
        datas_group = datas.groupby('vesseIMMSI')
        # 将同一艘船的订单存储到一个文件中
        for key, data in datas_group:
            path = os.path.join(output_path, name, key + '.csv')
            data.reset_index(drop=True, inplace=True)
            data.to_csv(path, index=None)

def stat():
    '''
    统计船个数
    :return:
    '''
    data_path = r'D:\baiph\BDC2020\data\train\preprocess'
    train_dirs = os.listdir(data_path)
    MMSIS = set()
    for dir in train_dirs:
        mmsis = os.listdir(os.path.join(data_path, dir))
        for mmsi in mmsis:
            mmsi = mmsi.split('.')[0]
            MMSIS.add(mmsi)
    MMSIS = list(MMSIS)
    MMSIS.sort()
    return MMSIS


def merge_csv_MMSI():
    '''
    根据MMSI合并文件,并去重
    :return:
    '''
    data_path = r'D:\baiph\BDC2020\data\train\preprocess/train%.3d'
    output_path = r'D:\baiph\BDC2020\data\train\train'
    MMSIS = stat()
    lines = 0
    for mm in tqdm(MMSIS):
        datas = []
        for i in range(1, 147):
            path = os.path.join(data_path % i, mm + '.csv')
            try:
                data = pd.read_csv(path)
                datas.append(data)
            except:
                pass
        datas = pd.concat(datas, axis=0)
        datas.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        length1 = datas.shape[0]
        datas.drop_duplicates(inplace=True)
        lines += (length1 - datas.shape[0])
        datas.reset_index(inplace=True, drop=True)
        datas.to_csv(os.path.join(output_path, mm + '.csv'), index=None)
    print('一共删除', str(lines), '行重复值')



def save_loadingOrder():
    '''
    由于一个订单的所有GPS数据不一定在一个船上，因此对不同船上同一订单进行合并处理
    '''
    path = r'../new-data/train'  # 以船为文件的路径
    out_path = r'../new-data/loadingOrders'

    files = os.listdir(path)
    loadingOrders = []
    multi_loadings = set()
    for file in tqdm(files):
        datas = pd.read_csv(os.path.join(path, file))
        datas = datas.sort_values(['loadingOrder', 'timestamp'])
        groups = datas.groupby('loadingOrder')
        for key, group in groups:
            group.reset_index(drop=True, inplace=True)
            if key not in loadingOrders:
                group.to_csv(os.path.join(out_path, key + '.csv'), index=None)
                loadingOrders.append(key)
            else:
                temp1 = pd.read_csv(os.path.join(out_path, key + '.csv'))
                temp1 = pd.concat([temp1, group], axis=0)
                temp1.sort_values(['loadingOrder', 'timestamp'], inplace=True)
                temp1.to_csv(os.path.join(out_path, key + '.csv'), index=None)
                multi_loadings.add(key)


if __name__ == "__main__":
    split_train_data()
    to_data_MMSI()
    merge_csv_MMSI()
    pass