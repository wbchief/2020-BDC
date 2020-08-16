import pandas as pd
import numpy as np


def score1(df, drop=True):
    '''
    官方评价标准
    ARRIVAL AT PORT实际到达目的港MYTPP为2019/09/18 13:28:46；选手预测为2019/09/18 22:28:46；
    目的港MYTPP的hETA1为(2019/09/18 22:28:46) – (2019/09/05 16:33:17) = 317.9925 （单位：小时），
    hATA1为(2019/09/18 13:28:4) – (2019/09/05 16:33:17) = 308.9925 （单位：小时）。
    :param df:
    :return:
    '''
    if drop:
        df.drop_duplicates(subset='loadingOrder', keep='first', inplace=True)

    num = df.shape[0]
    #print(df[['mmin', 'real', 'ETA']].head())
    df['real'] = pd.to_datetime(df['real'])
    df['ETA'] = pd.to_datetime(df['ETA'])
    #print(df[['mmin', 'real', 'ETA']].head())
    real_start = (df['real'] - pd.to_datetime(df['onboardDate']))
    pre_start = df['ETA'] - pd.to_datetime(df['onboardDate'])
    hETA1 = real_start.dt.total_seconds().values / 3600
    hATA1 = pre_start.dt.total_seconds().values / 3600
    #MSE = np.sum(np.square(hETA1 - hATA1, 2)) / num
    MSE = np.sum(((hETA1 - hATA1) * (hETA1 - hATA1))) / num
    # print(hETA1)
    # print(hATA1)

    return MSE

def get_data(data, mode='train'):
    assert mode == 'train' or mode == 'test'

    if mode == 'train':
        try:
            data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)

        except:
            pass
        data.rename(columns={"vesseIMMSI": 'vesselMMSI'}, inplace=True)
    elif mode == 'test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)

    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)
    return data

def submit(test_data, result, save_path, flag):
    '''
    生成提交csv
    :return:
    '''
    test_data = test_data.merge(result, on='loadingOrder', how='left')

    if flag:
        test_data1 = pd.read_csv(r'D:\baiph\BDC2020\data\test/R2 ATest 0711-1.csv')
        test_data1 = get_data(test_data1, 'test')
    #test_data
    # 按照第一个还是最后一个
    if not False:
        groups = test_data.groupby('loadingOrder')
        for key, group in groups:
            group.reset_index(drop=True, inplace=True)
            time = pd.to_datetime(group['timestamp'].values[-1])
            test_data.loc[test_data.loadingOrder == key, 'onboardDate'] = time

    test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
        lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)

    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    test_data['timestamp'] = test_data['temp_timestamp']

    if flag:
        test_data.drop_duplicates(subset=['loadingOrder'], keep='first', inplace=True)
        test_data1.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
        test_data1['onboardDate'] = test_data1['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
        test_data1['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
        test_data1['timestamp'] = test_data1['temp_timestamp']
        test_data = test_data1.merge(test_data[['loadingOrder', 'ETA']], on='loadingOrder', how='left')

    # 整理columns顺序
    result = test_data[
        ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
         'creatDate']]
    result.to_csv(save_path, index=False)


def reduce_mem_usage(props):
    '''减少内存的方法'''
    # 计算当前内存
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")

    # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
    NAlist = []
    for col in props.columns:
        # 这里只过滤了objectd格式，如果你的代码中还包含其他类型，请一并过滤
        if (props[col].dtypes != object):

            print("**************************")
            print("columns: ", col)
            print("dtype before", props[col].dtype)

            # 判断是否是int类型
            isInt = False
            mmax = props[col].max()
            mmin = props[col].min()

            # # Integer does not support NA, therefore Na needs to be filled
            # if not np.isfinite(props[col]).all():
            #     NAlist.append(col)
            #     props[col].fillna(-999, inplace=True) # 用-999填充

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = np.fabs(props[col] - asint)
            result = result.sum()
            if result < 0.01:  # 绝对误差和小于0.01认为可以转换的，要根据task修改
                isInt = True

            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0:  # 最小值大于0，转换成无符号整型
                    if mmax <= 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mmax <= 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mmax <= 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:  # 转换成有符号整型
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
            else:  # 注意：这里对于float都转换成float16，需要根据你的情况自己更改
                props[col] = props[col].astype(np.float16)

            print("dtype after", props[col].dtype)
            print("********************************")
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props  # , NAlist


def cal_distance(LatA,LatB,LonA,LonB):
    # 计算机两个经纬度之间的距离
    EARTH_RADIUS = 6378.137  # 千米
    def rad(d):
        return d * np.pi / 180.0
    s=0
    radLatA = rad(LatA)
    radLatB = rad(LatB)
    a = radLatA-radLatB
    b = rad(LonA)-rad(LonB)
    s= 2 * np.arcsin(np.sqrt(np.power(np.sin(a / 2),2)+ np.cos(radLatA) * np.cos(radLatB)*np.power(np.sin(b / 2),2)))
    s=s* EARTH_RADIUS
    #  保留两位小数
    s = np.round(s * 100)/100
    s = s * 1000  # 转换成m
    return s



def merge_result(path1, w1, path2, w2, save_path=None):
    '''
    模型融合
    :param path1: 模型1文件
    :param w1: 权重
    :param path2: 模型2文件
    :param w2: 权重
    :param save_path: 融合后的结果
    :return:
    '''
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    a1 = (pd.to_datetime(data1['ETA']) - pd.to_datetime(data1['onboardDate'])).dt.total_seconds().values
    a2 = (pd.to_datetime(data2['ETA']) - pd.to_datetime(data2['onboardDate'])).dt.total_seconds().values
    data = data1[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate']]
    data['label'] = w1 * a1 + w2 * a2
    data['ETA'] = (pd.to_datetime(data['onboardDate']) + data['label'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
        lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))

    data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    # 整理columns顺序
    result = data[
        ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
         'creatDate']]

    result.to_csv(save_path, index=False)

