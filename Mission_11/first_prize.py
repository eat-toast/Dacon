# -*- coding: utf8-*-
# General Library
import numpy as np
import pandas as pd
import time
import datetime
import random
import os
import matplotlib.pyplot as plt

# Machine Learning Library
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split


# Set Random Seed
seed = 777
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

# Print Information
print('Seed: %i'%(seed))
print('Numpy: %s'%(np.__version__))
print('Pandas: %s'%(pd.__version__))
print('LightGBM: %s'%(lgb.__version__))
print('Scikit-Learn: %s'%(sklearn.__version__))



def sort_dataset(dataset):
    '''
    This function sorts the meteric_id of train.csv and test.csv into numerical order.
    '''
    columns = dataset.columns
    meter_ids = columns[1:]
    tmp = []
    for meter_id in meter_ids:
        meter_id = meter_id.replace('X', '')
        meter_id = meter_id.replace('N', '') # 2020-03-14 컬럼이 기존 X1, X2 에서 NX1, NX2로 되어있어서 추가
        tmp.append(int(meter_id))
    tmp = np.sort(tmp)

    meter_ids = []
    for meter_id in tmp:
        # meter_id = 'X' + str(meter_id)
        meter_id = 'NX' + str(meter_id) # 2020-03-14 컬럼명이 변경 됨
        meter_ids.append(meter_id)

    results = [dataset[columns[0]].values]
    for meter_id in meter_ids:
        values = dataset[meter_id].values
        results.append(values)
    results = np.array(results).T
    df = pd.DataFrame(results, columns=[columns[0]] + meter_ids)
    return df


# set directory
os.chdir(os.path.join(os.getcwd(), 'Mission_11'))# 디렉토리 변경
data_path =  os.path.join(os.getcwd(), 'data')


# Load data
# train.csv, test.csv 파일이 metric id 순으로 정렬되어 있지 않아 이를 정렬하여 사용합니다.
# 2020-03-14 다운로드 받은 데이터는 모두 정렬이 되어 있다.
train = sort_dataset(pd.read_csv(os.path.join(data_path,'train.csv')))
test = sort_dataset(pd.read_csv(os.path.join(data_path,'test.csv')))


# Weather data
# '인천시간별기상자료(16-18)_축소_7월1일.csv' 파일은 인코딩 문제가 있어 텍스트로 불러옵니다.
with open(os.path.join(data_path,'인천_시간별__기상자료(16-18)_축소__7월1일.csv'), encoding = 'euc-kr') as file:
    additional = []
    for line in file.readlines():
        line = line.replace(',\n', ',nan')
        line = line.replace('\n', '')
        line = line.replace('뇌우끝,비', '뇌우끝_비')
        line = line.replace('뇌우,비눈', '뇌우_비눈')
        line = line.replace('뇌우끝,눈', '뇌우끝_눈')
        line = line.replace(',,', ',nan,')
        line = line.replace(',,', ',nan,')
        additional.append(line.split(','))
# 추가 데이터 정리
additional = np.array(additional)
additional_columns = additional[0]
additional_datas = additional[1:]
additional_datas_float = []

# 모두 string으로 저장되어 있어, column 별로 타입을 변환하려 한다.
for i in range(len(additional_columns)):
    additional_data = additional_datas[:, i]
    try:
        # float 변환이 된다면, 변환
        tmp = additional_data.astype(float)
        additional_datas_float.append(tmp)
    except:
        # 안된다면, strinf 타입으로 가져오기
        tmp = additional_data.astype(str)
        additional_datas_float.append(tmp)

# 추가 데이터 일자가 저장되어 있는 곳
add_times = []
for item in additional_datas_float[1]:
    new_time = datetime.datetime.strptime(item, '%Y.%m.%d %H:%M')
    add_times.append(new_time)
add_times = np.array(add_times)

# 추가 데이터 기온이 저장되어 있는 곳
temperature = additional_datas_float[2]
for i in range(len(temperature)):
    if np.isnan(temperature[i]):
        print(i)
        temperature[i] = temperature[i-1] # nan 이라면, 이전시간의 온도를 가져온다.

# 추가 데이터 강수량
rainfall = additional_datas_float[3]
rainfall[np.where(np.isnan(rainfall) == True)[0]] = 0 # nan이라면, 0을 삽입

# 추가 데이터 풍속
wind = additional_datas_float[4]

# 추가 데이터 습도
humidity = additional_datas_float[5]
for i in range(len(humidity)):
    if np.isnan(humidity[i]):
        humidity[i] = humidity[i-1] # nan 이라면, 이전시간의 습도를 가져온다.

# 추가 데이터 적설량
snowfall = additional_datas_float[6]
snowfall[np.where(np.isnan(snowfall) == True)[0]] = 0 # nan이라면, 0을 삽입

# 추가 데이터 전운량
cloud = additional_datas_float[8]
for i in range(len(cloud)):
    if np.isnan(cloud[i]):
        cloud[i] = cloud[i-1] # nan 이라면, 이전시간의 전운량을 가져온다.


# 2. 데이터 전처리
# Data Cleansing & Pre-Processing
# 1) train, test의 결측치를 모두 0으로 변경합니다.
# 2) 날짜의 타입을 datetime 으로 변경합니다.
# 3) 시간별 날씨 데이터를 평균을 이용하여 일자별 날씨로 변경합니다.
def time_convert(df_time, string_type='train'):
    '''
    This function changes format of time from string to datetime.
    '''
    old_times = df_time
    new_times = old_times.copy()
    for i, old_time in enumerate(old_times):
        if string_type == 'train':
            new_time = datetime.datetime.strptime(old_time, '%Y-%m-%d %H:%M')
        elif string_type == 'test':
            new_time = datetime.datetime.strptime(old_time, '%Y.%m.%d %H:%M')
        else:
            new_time = datetime.datetime.strptime(old_time, '%Y-%m-%d')
        new_times[i] = new_time
    return new_times


def split_day(_times, _datas):
    '''
    This function splits power consumption data and weather data by days.
    '''
    for time in _times:
        if time.time().hour == 0: # 첫 0 시 일자
            ref_time = time.date()
            break
    times = []
    datas = []
    data_tmp = []

    for i, time in enumerate(_times):
        time = time.date()
        data = _datas[i]

        if ref_time > time: # ref_time &gt; time:
            pass

        elif ref_time == time:
            data_tmp.append(data)

        else:
            times.append(ref_time) # 당일 시간 정보는 times로 모으
            datas.append(data_tmp) # 당일 데이터는 datas로 모으기
            ref_time = time # 다음 시간으로 변경
            data_tmp = [data] # 전날 00시 데이터로 초기화

    if ref_time not in times: # 마지막 ref_time이 times에 포함되어 있지 않다면
        if len(data_tmp) == 24: # data의 갯수가 24개일 때, 포함시킨다.
            times.append(ref_time)
            datas.append(data_tmp)
    times = np.array(times)
    datas = np.array(datas)
    return times, datas


# Repalce nan to zero
train = train.replace(np.nan, 0.0)
test = test.replace(np.nan, 0.0)
# Convert time data format to datetime
train_times = time_convert(train['Time'], string_type='train').values
test_times = time_convert(test['Time'], string_type='test').values
# Meter id
train_meter_ids = train.columns[1:]
test_meter_ids = test.columns[1:]




tt = pd.DataFrame(data = {'add_times': add_times,
                     'temperature':temperature,
                     'rainfall':rainfall,
                     'wind':wind,
                     'humidity': humidity,
                     'snowfall': snowfall,
                     'cloud':cloud
                     })


# Downsampling (a day)
temperature = np.mean(split_day(add_times, temperature)[1], axis=1)
rainfall = np.mean(split_day(add_times, rainfall)[1], axis=1)
wind = np.mean(split_day(add_times, wind)[1], axis=1)
humidity = np.mean(split_day(add_times, humidity)[1], axis=1)
snowfall = np.mean(split_day(add_times, snowfall)[1], axis=1)
#add_times, cloud = split_day(add_times, cloud)
#cloud = np.mean(cloud, axis=1)

tt['date'] = [x.date() for x in tt.add_times]

tt2 = tt.groupby('date')['temperature'].mean()

# Make additional data set
additional_data = np.array([
    add_times,
    temperature,
    rainfall,
    wind,
    humidity,
    snowfall,
    cloud,
])


# 3. 탐색적 자료분석
# Exploratory Data Analysis
# 아래 그림은 train.csv, test.csv 내 유효한 데이터 수를 보여줍니다. test.csv는 id 481번 이하로 분포하고 있으며 481번을 기준으로 데이터 수의 분포가 다릅니다.
# 따라서 모델 생성 시 0 ~ 481번 데이터만 사용하는 것을 고려합니다.

train_id_num = []
train_data_num = []
for meter_id in train_meter_ids:
    meter_num = int(meter_id.replace('NX', '')) # 데이터가 몇번째 컬럼에 위치하는지 확인
    valid_num = len(np.where(train[meter_id] > 0.0)[0]) # 0초과 데이터만 갯수 확인
    train_id_num.append(meter_num)
    train_data_num.append(valid_num)

test_id_num = []
test_data_num = []
for meter_id in test_meter_ids:
    meter_num = int(meter_id.replace('NX', ''))
    valid_num = len(np.where(test[meter_id] > 0.0)[0])
    test_id_num.append(meter_num)
    test_data_num.append(valid_num)

plt.scatter(train_id_num, train_data_num, s=5, color='b', label='train')
plt.scatter(test_id_num, test_data_num, s=5, color='r', label='test')
plt.legend()
plt.xlim(0, 1500)
plt.ylim(bottom=0)
plt.xlabel('Meter Id')
plt.ylabel('Data Numer')
plt.show()



