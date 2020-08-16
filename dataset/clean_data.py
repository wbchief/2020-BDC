import os
import random

from tqdm import tqdm
import pandas as pd

from utils import cal_distance


def clean_train_data():
    '''
    清洗数据
    :return:
    '''
    # train_gps_path = r'D:\baiph\BDC2020\data\train\train'
    # out_path = r'D:\baiph\BDC2020\data\train\data\train-clean'
    train_gps_path = r'D:\baiph\BDC2020\data\train\cleans\clean1'

    out_path = r'D:\baiph\BDC2020\data\train\cleans\clean1-train'
    mmsis = os.listdir(train_gps_path)
    thread = 30000
    for mm in tqdm(mmsis):
        path = os.path.join(train_gps_path, mm)
        datas = pd.read_csv(path)
        datas['timestamp'] = pd.to_datetime(datas['timestamp'], infer_datetime_format=True)
        datas.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        groups = datas.groupby('loadingOrder')

        result = []
        for key, group in groups:
            group = group.reset_index(drop=True)
            # 清洗起航时数据，将speed为0的删除
            id = group['speed'].ne(0).idxmax()
            group = group.iloc[id:]
            group.reset_index(drop=True, inplace=True)

            last_address = group[['longitude', 'latitude']].values[-1]
            address = group[['longitude', 'latitude']].values
            count = 0
            for i, add in enumerate(address):
                # 清洗到港数据
                a = cal_distance(last_address[1], add[1], last_address[0], add[0])
                if a < thread and group['speed'].iloc[i] == 0:
                    count = i
                    break
            if count != 0:
                group = group.iloc[0:count]
            group.reset_index(drop=True, inplace=True)
            result.append(group)
        if len(result) != 0:

            result = pd.concat(result, axis=0)
            result.to_csv(os.path.join(out_path, mm), index=None)


def find_similar_data():
    '''
    相似数据：指目的港口数据在阈值内
    :return:
    '''
    # input_path = r'D:\baiph\BDC2020\data\train\data\train-clean'
    # out_path = r'D:\baiph\BDC2020\data\train\data\similar\train'
    input_path = r'D:\baiph\BDC2020\data\train\cleans\clean2'
    out_path = r'D:\baiph\BDC2020\data\train\cleans\silimar2'
    files = os.listdir(input_path)

    test_path = r'D:\baiph\BDC2020\data\test/testA.csv'
    test_data = pd.read_csv(test_path).groupby('loadingOrder')


    test_gloal_port = []
    for key, group in test_data:
        test_gloal_port.append(group[['end_longitude', 'end_latitude']].values[0])

    thread = 30000
    for file in tqdm(files):

        datas = pd.read_csv(os.path.join(input_path, file)).groupby('loadingOrder')
        for key, data in datas:
            data.reset_index(drop=True, inplace=True)
            gloal_port = data[['longitude', 'latitude']].values[-1]
            for port in test_gloal_port:
                if cal_distance(gloal_port[1], port[1], gloal_port[0], port[0]) < thread:
                    data.to_csv(os.path.join(out_path, key + '.csv'), index=None)
                    break


def struct_data(input_path, out_path):
    '''构造相似数据训练集'''

    files = os.listdir(input_path)
    # input_path = r'D:\baiph\BDC2020\data\train\cleans\clean1-train'
    # out_path = r'D:\baiph\BDC2020\data\train\cleans\struct1'

    columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude', 'vesseIMMSI', 'speed',
               'direction', 'start_lat', 'end_lat', 'start_lon', 'end_lon', 'onboardDate', 'label']
    for file in tqdm(files):
        datas = pd.read_csv(os.path.join(input_path, file))
        datas['timestamp'] = pd.to_datetime(datas['timestamp'], infer_datetime_format=True)
        # datas.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        # datas = datas.reset_index(drop=True)
        start_lat, end_lat = datas['latitude'].values[0], datas['latitude'].values[-1]
        start_lon, end_lon = datas['longitude'].values[0], datas['longitude'].values[-1]
        datas['start_lat'] = start_lat
        datas['end_lat'] = end_lat
        datas['start_lon'] = start_lon
        datas['end_lon'] = end_lon

        length = datas.shape[0]
        try:
            random1 = random.uniform(0, 1)
            if random1 < 0.5:
                data = datas.iloc[0:int(length * random.uniform(0.1, 0.5))]
            elif random1 > 0.5 and random1 < 0.8:
                data = datas.iloc[0:int(length * random.uniform(0.4, 0.8))]
            else:
                data = datas.iloc[0:int(length * random.uniform(0.7, 0.95))]
            data = data.sample(frac=0.5, replace=False, random_state=1)
            data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['onboardDate'] = data['timestamp'].iloc[-1]
            data['label'] = (datas['timestamp'].iloc[-1] - data['timestamp'].iloc[-1]).total_seconds() / 3600
            data.to_csv(os.path.join(out_path, file), index=None)
        except:
            pass


def struct_data1(input_path, out_path):
    '''构造数据训练集'''

    files = os.listdir(input_path)
    for file in tqdm(files):
        datas = pd.read_csv(os.path.join(input_path, file))
        datas['timestamp'] = pd.to_datetime(datas['timestamp'], infer_datetime_format=True)
        groups = datas.groupby('loadingOrder')
        result = []
        for key, group in groups:
            group.reset_index(inplace=True, drop=True)
            start_lat, end_lat = group['latitude'].values[0], group['latitude'].values[-1]
            start_lon, end_lon = group['longitude'].values[0], group['longitude'].values[-1]
            group['start_lat'] = start_lat
            group['end_lat'] = end_lat
            group['start_lon'] = start_lon
            group['end_lon'] = end_lon

            length = group.shape[0]
            try:
                data = group.iloc[0:int(length * random.uniform(0.2, 0.7))]
                data = data.sample(frac=0.45, replace=False, random_state=1)
                data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
                data.reset_index(drop=True, inplace=True)
                data['onboardDate'] = data['timestamp'].iloc[-1]
                data['label'] = (group['timestamp'].iloc[-1] - data['timestamp'].iloc[-1]).total_seconds() / 3600
                result.append(data)
            except:
                pass

        if len(result) == 0:
            pass
        else:
            result = pd.concat(result, axis=0)
            result.to_csv(os.path.join(out_path, file), index=None)

if __name__ == '__main__':
    # clean_train_data()
    #find_similar_data()
    # 相似轨迹
    # clean_train_data()
    struct_data(r'D:\baiph\BDC2020\data\train\cleans\clean2',
                r'D:\baiph\BDC2020\data\train\cleans\struct2')

    # struct_data1(r'D:\baiph\BDC2020\data\train\data\train-clean',
    #             r'D:\baiph\BDC2020\data\train\data\all\struct-train')
    pass